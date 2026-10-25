"""AWQ 实现冒烟测试（小张量，不加载大模型；板上可随时重跑）。

  taskset -c 0-3 .venv/bin/python scripts/awq_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)

from src.nac.awq import (apply_fold, awq_quant_weight, chan_weights,  # noqa: E402
                         expand_s_o, fold_block_, undo_fold_norms, _grid_search)
from src.nac.quantize import dequantize_weight, quantize_weight  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        fails.append(name)


# ---------- 1) expand_s_o 轨道映射 ----------
n_kv, hd, gpd = 2, 8, 3
sigma = torch.rand(n_kv * hd) + 0.5
se = expand_s_o(sigma, n_kv, hd, gpd)
ok_map = all(abs(se[(g * gpd + k) * hd + d].item() - sigma[g * hd + d].item()) < 1e-7
              for g in range(n_kv) for k in range(gpd) for d in range(hd))
check("expand_s_o orbit mapping", ok_map and se.shape == (n_kv * gpd * hd,))

# ---------- 2) o/v 折叠经假注意力（头复制 E）等价 ----------
V = torch.randn(n_kv * hd, 32) * 0.1
O = torch.randn(20, n_kv * gpd * hd) * 0.1
x = torch.randn(32)
E = torch.zeros(n_kv * gpd * hd, n_kv * hd)
for g in range(n_kv):
    for k in range(gpd):
        for d in range(hd):
            E[(g * gpd + k) * hd + d, g * hd + d] = 1.0
y0 = O @ (E @ (V @ x))
Vf = V / sigma.unsqueeze(1)                      # v 行 ÷ σ
Of = O * expand_s_o(sigma, n_kv, hd, gpd)        # o 列 × expand(σ)
y1 = Of @ (E @ (Vf @ x))
check("o_proj GQA orbit fold equivalence", torch.allclose(y0, y1, atol=1e-5),
      f"max|dy|={ (y0-y1).abs().max():.2e}")

# ---------- 3) norm 折叠等价（RMSNorm 逐通道） ----------
w = torch.randn(32) * 0.5 + 1.0
W = torch.randn(40, 32) * 0.1
s = torch.rand(32) + 0.5
xn = torch.randn(8, 32)                          # 已归一化激活
y0 = (w * xn) @ W.T
y1 = ((w / s) * xn) @ (W * s).T
check("RMSNorm fold equivalence", torch.allclose(y0, y1, atol=1e-5))

# ---------- 4) mock block 全链折叠等价（fold_block_ 路径） ----------
import torch.nn as nn  # noqa: E402


class MockAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 40, bias=False)
        self.k_proj = nn.Linear(32, n_kv * hd, bias=False)
        self.v_proj = nn.Linear(32, n_kv * hd, bias=False)
        self.o_proj = nn.Linear(n_kv * gpd * hd, 24, bias=False)


class MockMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 64, bias=False)
        self.up_proj = nn.Linear(32, 64, bias=False)
        self.down_proj = nn.Linear(64, 24, bias=False)


class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Linear(32, 32, bias=False)      # 仅用 weight 当 norm 权重
        self.post_attention_layernorm = nn.Linear(32, 32, bias=False)
        self.self_attn, self.mlp = MockAttn(), MockMlp()


blk = MockBlock()
for m in blk.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.1)
blk.input_layernorm.weight.data = torch.rand(32) + 0.5
blk.post_attention_layernorm.weight.data = torch.rand(32) + 0.5


def block_out(b, x):
    xn1 = b.input_layernorm.weight * x
    q = b.self_attn.q_proj(xn1)
    v = E @ b.self_attn.v_proj(xn1)               # 头复制（假注意力，P 与 V 缩放对易）
    attn_out = b.self_attn.o_proj(v)
    xn2 = b.post_attention_layernorm.weight * x
    h = torch.nn.functional.silu(b.mlp.gate_proj(xn2)) * b.mlp.up_proj(xn2)
    return q.sum() + attn_out.sum() + b.mlp.down_proj(h)


xs = torch.randn(32)
with torch.no_grad():
    y0 = block_out(blk, xs)
    sa, so, sm, sd = (torch.rand(32) + 0.5, torch.rand(n_kv * hd) + 0.5,
                      torch.rand(32) + 0.5, torch.rand(64) + 0.5)
    fold_block_(blk, sa, so, sm, sd, head_dim=hd)
    y1 = block_out(blk, xs)
check("mock block fold equivalence (fold_block_)", torch.allclose(y0, y1, rtol=1e-4, atol=1e-5),
      f"rel={((y0-y1).abs()/y0.abs()).max():.2e}")

# apply_fold / undo_fold_norms（model.model.layers 接口）
with torch.no_grad():
    w_norm0 = blk.input_layernorm.weight.data.clone()
    w_post0 = blk.post_attention_layernorm.weight.data.clone()
    mock_model = type("M", (), {"model": type("L", (), {"layers": [blk]})()})()
    fold_dict = {0: {"s_attn": sa, "s_o": so, "s_mlp": sm, "s_down": sd}}
    apply_fold(mock_model, fold_dict, head_dim=hd)
    check("apply_fold reproduces calibration fold", torch.allclose(
        blk.input_layernorm.weight.data, w_norm0 / sa, atol=1e-7))
    undo_fold_norms(mock_model, fold_dict)
    check("undo_fold_norms restores norms", torch.allclose(
        blk.input_layernorm.weight.data, w_norm0, atol=1e-6) and
        torch.allclose(blk.post_attention_layernorm.weight.data, w_post0, atol=1e-6))

# ---------- 5) int-zp 整格点性质 ----------
W = torch.randn(64, 128) * 0.05
cw = torch.rand(128) + 0.1
scale, zp = awq_quant_weight(W, 64, cw)
check("int-zp integer valued", bool((zp == torch.round(zp)).all()))
res = quantize_weight(W, 64, scale, zp)
dq = dequantize_weight(res.q, res.scale, res.zp).reshape(64, -1, 64)
# dq = scale*(q-zp) 定义上在 scale*Z 格点；除回 scale 仅剩 fp32 舍入（~1e-6）
grid_err = float((dq / scale.unsqueeze(-1) + zp.unsqueeze(-1) - res.q).abs().max())
check("dequant on scale*Z grid", grid_err < 1e-4, f"fp round-off={grid_err:.2e}")

# ---------- 6) AWQ 加权误差 <= minmax 基线 ----------
gs = 64
for seed in range(5):
    torch.manual_seed(seed)
    Wt = torch.randn(256, 256) * 0.05
    xb = torch.rand(256) + 0.1
    sc, zpb = awq_quant_weight(Wt, gs, xb.pow(2))         # 基线：s=1（无折叠），加权 minmax+clip
    r = quantize_weight(Wt, gs, sc, zpb)
    dqb = dequantize_weight(r.q, r.scale, r.zp)
    err_mm = float(((dqb - Wt).pow(2) * xb.pow(2)).sum())
    s_best, a_best = _grid_search(xb, Wt.abs().amax(0), [Wt], gs)
    sc2, zp2 = awq_quant_weight(Wt * s_best, gs, chan_weights(xb, s_best))
    r2 = quantize_weight(Wt * s_best, gs, sc2, zp2)
    dq2 = dequantize_weight(r2.q, r2.scale, r2.zp)
    err_awq = float(((dq2 - Wt * s_best).pow(2) * (xb / s_best).pow(2)).sum())
    check(f"awq<=minmax seed{seed}", err_awq <= err_mm,
          f"a*={a_best} gain={(1 - err_awq / err_mm) * 100:.1f}%")

print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
