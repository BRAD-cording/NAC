"""NAC-v2 量化核心：Hessian 列序误差补偿 + 激活加权组参数搜索。

问题 A（docs/review_publication_20261017.md §3）修复：
NAC grid 的目标 Σ a⊙(W−deq(W))² 是纯权重空间组内寻优，无列间误差扩散，
长程算术（GSM8K CoT）上误差累积崩溃（sim 0.070 vs gptq 0.5625，10-17 E2）。

v2 = GPTQ 列序 Cholesky 补偿骨架（误差扩散进后续列）+ NAC 激活加权 κ 网格组搜索：
- 组搜索在"当前已补偿权重"上进行（与 GPTQ 相同的时序语义）
- 激活权重取 H 对角（token 二阶矩），与输出空间加权 MSE 目标一致；
  GPTQ 基线的 maxshrink 网格是均匀加权特例，v2 在同一补偿骨架上加权寻优
- int_zp=True（默认）：zp 限制到整格点，同时服务问题 B
  （RK3576 NPU 对 float zp 的取整/重解释嫌疑，E3 裁决）

NPU 反量化语义与 quantize.py 一致：W = scale * (q - zp)，q ∈ [0,15] uint4。
"""
import torch

from .quantize import QMAX, QMIN

DEFAULT_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4)
PERCDAMP = 0.01
BLOCKSIZE = 128


def find_params_group_aw(wg: torch.Tensor, aw: torch.Tensor,
                         ratios=DEFAULT_RATIOS, int_zp: bool = True):
    """激活加权 κ 网格组参数搜索。

    wg: (out, g) 当前已补偿组权重；aw: (g,) 组内通道激活权重（diag(H)，均值归一）。
    返回 (scale, zp) 形状 (out,)。int_zp 时 zp 取整（整值浮点，Rockchip 语义）。
    """
    wmin, wmax = wg.amin(-1), wg.amax(-1)
    wmax = torch.maximum(wmax, wmin + 1e-8)
    best_s = best_z = best_err = None
    for r in ratios:
        span = (wmax - wmin) * r
        center = (wmax + wmin) / 2
        lo = torch.maximum(center - span / 2, wmin)
        hi = torch.minimum(center + span / 2, wmax)
        hi = torch.maximum(hi, lo + 1e-8)
        scale = (hi - lo) / (QMAX - QMIN)
        zp = -lo / scale
        if int_zp:
            zp = torch.round(zp)
        q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + zp.unsqueeze(-1)),
                        QMIN, QMAX)
        err = (((scale.unsqueeze(-1) * (q - zp.unsqueeze(-1)) - wg) ** 2) * aw).sum(-1)
        if best_err is None:
            best_s, best_z, best_err = scale, zp, err
        else:
            better = err < best_err
            best_s = torch.where(better, scale, best_s)
            best_z = torch.where(better, zp, best_z)
            best_err = torch.where(better, err, best_err)
    return best_s, best_z


def gptq_aw_linear(W: torch.Tensor, H: torch.Tensor, gs: int,
                   aw: torch.Tensor | None = None,
                   ratios=DEFAULT_RATIOS, int_zp: bool = True):
    """GPTQ 列序 Hessian 补偿 + 激活加权组搜索（v2 主核）。

    W: (out, in) fp32，就地改写为反量化权重；
    H: (in, in) fp32 平均 Hessian（尺度不影响解）；
    aw: (in,) 通道激活权重，None 则均匀（退化为准 GPTQ）。
    返回 (scale_all, zp_all) 形状 (out, in/gs)。
    """
    out_f, in_f = W.shape
    if aw is None:
        aw = torch.ones(in_f)
    else:
        aw = aw.clone().float()
        aw = aw / aw.mean().clamp_min(1e-12)

    dead = torch.diag(H) == 0
    H = H.clone()
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = PERCDAMP * torch.mean(torch.diag(H))
    H[range(in_f), range(in_f)] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)

    ng = in_f // gs
    scale_all = torch.zeros(out_f, ng)
    zp_all = torch.zeros(out_f, ng)
    for i1 in range(0, in_f, BLOCKSIZE):
        i2 = min(i1 + BLOCKSIZE, in_f)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            col = i1 + i
            if col % gs == 0:
                sc, zp = find_params_group_aw(W[:, col: col + gs], aw[col: col + gs],
                                              ratios=ratios, int_zp=int_zp)
                scale_all[:, col // gs] = sc
                zp_all[:, col // gs] = zp
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale_all[:, col // gs]) + zp_all[:, col // gs],
                            QMIN, QMAX)
            dq = scale_all[:, col // gs] * (q - zp_all[:, col // gs])
            Q1[:, i] = q
            W1[:, i] = dq
            err = (w - dq) / d
            if i + 1 < count:
                W1[:, i + 1:] -= err.unsqueeze(1) @ Hinv1[i, i + 1:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i1:i2] = W1
        if i2 < in_f:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return scale_all, zp_all
