"""AWQ 基线（P0-1 扩展：Lin et al. 2023, AWQ: Activation-aware Weight
Quantization）项目内实现，量化框架与 NAC/GPTQ 严格一致（非对称 uint4、
per-group、NPU 公式 W = scale*(q - zp)、整值 zp）。

核心（对应论文 Method 的 AWQ 基线小节）：
- 激活感知 scale：s = s_x^α / s_w^(1-α)，α 网格 0-1（21 点）最小化
  Σ_j (Δw'_j · x̄_j / s_j)²，w' = W·diag(s) 为折叠后权重，Δw' 为其
  伪量化误差（α 搜索用 minmax 轻量代理，最终参数用 clip 网格核）
- 共享输入联合搜索：q/k/v 同源 input_layernorm、gate/up 同源
  post_attention_layernorm → 组内单一 s 联合最小化（s_w 取成员均值）
- GQA o_proj：o 输入通道 j 的 v 源行 r(j) = (j//head_dim)//gpd*head_dim
  + j%head_dim（gpd = heads/kv_heads）；σ 按 v 行粒度（轨道）约束搜索，
  6 个头共享 v 行 → 轨道内 s 必须一致，折进 v_proj 权重行
- 折叠：RMSNorm / SiLU(gate)⊙up 均逐通道可乘 → 生产者权重 ÷ s（发射
  x/s）、消费者权重列 × s（量化 W·diag(s)），端到端等价变换
- fold.pt 只存 scale 向量（每块 ~1.2 万浮点，全模型 ~1.5MB），eval 端
  用同一 fold_block_ 重建折叠权重（与校准位级一致；不落 6GB 权重，
  避开板端 sim 评测期的 RAM 峰值）
"""
import torch

from src.nac.quantize import QMAX, QMIN

ALPHA_GRID = [round(i * 0.05, 2) for i in range(21)]  # 0.00-1.00
CLIP_GRID = 20        # p 网格点数（shrink 1.0 -> 0.8）
MAXSHRINK = 0.8
HEAD_DIM = 128        # Qwen2.5-1.5B-Instruct（12 heads / 2 kv heads / hd 128）


def expand_s_o(s_o: torch.Tensor, n_kv_heads: int, head_dim: int, gpd: int) -> torch.Tensor:
    """轨道 scale σ(v 行粒度, n_kv*hd) -> o 输入通道粒度 (n_kv*gpd*hd,)。

    j = (g*gpd + k)*hd + d -> σ[g*hd + d]（同 kv 组同偏移的 gpd 个头共享 v 行）。
    """
    return s_o.view(n_kv_heads, 1, head_dim).expand(n_kv_heads, gpd, head_dim).reshape(-1)


@torch.no_grad()
def fold_block_(block, s_attn, s_o, s_mlp, s_down, head_dim: int = HEAD_DIM):
    """块内激活感知折叠（等价变换；校准与 eval 复用同一实现保证位级一致）。

    生产者权重 ÷ s（使下游输入变为 x/s），消费者权重列 × s（量化目标
    W·diag(s)）。q/k/v 共享 s_attn；o 的 σ 折进 v_proj 行（GQA 轨道）；
    down 的 s_down 折进 up_proj 行（SiLU 逐通道合法）。
    """
    attn, mlp = block.self_attn, block.mlp
    n_kv_heads = attn.v_proj.weight.shape[0] // head_dim
    gpd = attn.o_proj.weight.shape[1] // (n_kv_heads * head_dim)

    block.input_layernorm.weight.div_(s_attn)
    for n in ("q_proj", "k_proj", "v_proj"):       # 消费者列 × s_attn
        getattr(attn, n).weight.mul_(s_attn)
    attn.v_proj.weight.div_(s_o.reshape(-1).unsqueeze(1))   # 生产者行 ÷ σ（发射 attn/σ）；σ 轨道形式 [n_kv,hd] 摊平为 v 行 [n_kv*hd]
    if attn.v_proj.bias is not None:               # 生产者 bias 同步 ÷σ（否则 A·b 项无法被 o 列乘补偿）
        attn.v_proj.bias.div_(s_o.reshape(-1))
    attn.o_proj.weight.mul_(expand_s_o(s_o, n_kv_heads, head_dim, gpd))
    block.post_attention_layernorm.weight.div_(s_mlp)
    mlp.gate_proj.weight.mul_(s_mlp)
    mlp.up_proj.weight.mul_(s_mlp)
    mlp.up_proj.weight.div_(s_down.unsqueeze(1))   # 生产者行 ÷ s_down
    if mlp.up_proj.bias is not None:               # 生产者 bias 同步 ÷s_down
        mlp.up_proj.bias.div_(s_down)
    mlp.down_proj.weight.mul_(s_down)


@torch.no_grad()
def apply_fold(model, fold: dict, head_dim: int = HEAD_DIM):
    """eval 用：按 fold 参数（awq_calibrate.py 产物）折叠全部块。"""
    for bi, block in enumerate(model.model.layers):
        f = fold[bi]
        fold_block_(block, f["s_attn"], f["s_o"], f["s_mlp"], f["s_down"],
                    head_dim=head_dim)


@torch.no_grad()
def undo_fold_norms(model, fold: dict):
    """撤销 norms 折叠（混合配置评测时恢复；linears 由 restore_from_disk 精确还原）。

    fold 对 norm 只有 ÷s 一种操作，逆操作 ×s 的 fp 舍入误差 ~ulp，可忽略。
    """
    for bi, block in enumerate(model.model.layers):
        f = fold[bi]
        block.input_layernorm.weight.mul_(f["s_attn"])
        block.post_attention_layernorm.weight.mul_(f["s_mlp"])


def _quant_err_map(w: torch.Tensor, gs: int) -> torch.Tensor:
    """per-group minmax 伪量化误差图 (out, in/g, gs)（α 搜索的轻量代理）。"""
    wg = w.reshape(w.shape[0], -1, gs)
    wmin, wmax = wg.amin(-1), wg.amax(-1)
    scale = ((wmax - wmin) / QMAX).clamp_min(1e-10)
    zero = torch.round(-wmin / scale)
    q = torch.clamp(torch.round(wg / scale.unsqueeze(-1)) + zero.unsqueeze(-1), QMIN, QMAX)
    return (scale.unsqueeze(-1) * (q - zero.unsqueeze(-1)) - wg).pow(2)


def _weighted_err(ws: list, s_full: torch.Tensor, x_bar: torch.Tensor,
                  gs: int) -> float:
    """Σ_members Σ_j (Δw'_j · x̄_j / s_j)²，s_full 为输入通道粒度 scale。"""
    cw = (x_bar / s_full).pow(2).reshape(1, -1, gs)
    tot = 0.0
    for w in ws:
        wf = w * s_full.unsqueeze(0)
        tot += float((_quant_err_map(wf, gs) * cw).sum())
    return tot


def _grid_search(x_bar: torch.Tensor, s_w: torch.Tensor, ws: list,
                 gs: int, orbit: tuple | None = None):
    """α 网格搜索激活感知 scale。

    orbit=None: 返回 s（输入通道粒度）；orbit=(n_kv, head_dim, gpd) 时先聚合到
    轨道粒度（v 行）搜索，返回 σ 与 alpha（调用方 expand）。
    """
    if orbit is not None:
        n_kv, head_dim, gpd = orbit
        agg = lambda t: t.reshape(n_kv, gpd, head_dim).mean(1)  # noqa: E731 轨道内取均值
        x_bar_e, s_w_e = agg(x_bar), agg(s_w)
    else:
        x_bar_e, s_w_e = x_bar, s_w

    best_s = best_a = best_e = None
    for a in ALPHA_GRID:
        s = (x_bar_e.clamp_min(1e-8) ** a) / (s_w_e.clamp_min(1e-8) ** (1 - a))
        s = s.clamp_min(1e-4)
        s_full = expand_s_o(s, n_kv, head_dim, gpd) if orbit is not None else s
        e = _weighted_err(ws, s_full, x_bar, gs)
        if best_e is None or e < best_e:
            best_s, best_a, best_e = s, a, e
    return best_s, best_a


def awq_quant_weight(w: torch.Tensor, gs: int, chan_w: torch.Tensor,
                     max_shrink: float = MAXSHRINK, grid: int = CLIP_GRID):
    """AWQ 量化核：折叠后权重 per-group clip 网格 + (x̄/s)² 加权误差。

    w: (out, in) 折叠后权重；chan_w: (in,) 逐通道误差权重。
    返回 scale, zp 形状 (out, in/g)；zp 整值浮点（NPU 公式 W=scale*(q-zp)
    的整格点域，对齐 GPTQ 基线与问题 B 的 int-zp 处置）。
    """
    wg = w.reshape(w.shape[0], -1, gs)
    cw = chan_w.reshape(1, -1, gs)
    wmin, wmax = wg.amin(-1), wg.amax(-1)
    best_scale = best_zp = best_err = None
    for i in range(grid + 1):
        p = 1 - i / grid * max_shrink
        xmin2, xmax2 = p * wmin, p * wmax
        scale = (xmax2 - xmin2).clamp_min(1e-10) / QMAX
        zero = torch.round(-xmin2 / scale)  # 整值 zp：q clamp 兜底（GPTQ 官方语义）
        q = torch.clamp(torch.round(wg / scale.unsqueeze(-1)) + zero.unsqueeze(-1),
                        QMIN, QMAX)
        err = (cw * (scale.unsqueeze(-1) * (q - zero.unsqueeze(-1)) - wg).pow(2)).sum(-1)
        if best_err is None:
            best_scale, best_zp, best_err = scale, zero, err
        else:
            better = err < best_err
            best_scale = torch.where(better, scale, best_scale)
            best_zp = torch.where(better, zero, best_zp)
            best_err = torch.where(better, err, best_err)
    return best_scale, best_zp


def chan_weights(x_bar: torch.Tensor, s_full: torch.Tensor) -> torch.Tensor:
    """量化核的逐通道误差权重 (x̄_j/s_j)²。"""
    return (x_bar / s_full).pow(2)
