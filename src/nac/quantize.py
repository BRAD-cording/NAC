"""W4A16 per-group 量化核。

Rockchip RK3576 NPU W4A16 反量化公式（方案贡献1）：
    W_dequant[i] = scale[g] * (W_int4[i] - zp[g]),  g = i // group_size

实现要点：
- 非对称 4bit（uint4 量化域 [0, 15]），per-group 独立 scale / zero-point
- 校准期使用 STE（straight-through estimator）让 scale/zp 可学习
- 提供 min-max 与 MSE 网格搜索两种初始化
"""
from dataclasses import dataclass

import torch

QMIN, QMAX = 0, 15  # uint4


@dataclass
class GroupQuantResult:
    q: torch.Tensor        # (out, in/g, g) int4-in-int8 域
    scale: torch.Tensor    # (num_groups, ) float
    zp: torch.Tensor       # (num_groups, ) float（软连续，部署时 round）


def _reshape_groups(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """(out, in) -> (out, in/g, g)。in 必须被 group_size 整除。"""
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by group {group_size}"
    return w.reshape(out_dim, in_dim // group_size, group_size)


def init_minmax(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """per-group min-max 初始化 scale/zp，返回 (scale, zp) 均为 (out, in/g) 形状。"""
    wg = _reshape_groups(w, group_size)
    wmin = wg.amin(dim=-1)
    wmax = wg.amax(dim=-1)
    # 防止空组
    wmax = torch.maximum(wmax, wmin + 1e-8)
    scale = (wmax - wmin) / (QMAX - QMIN)
    zp = -wmin / scale  # W=scale*(q-zp)：q=0 ↔ w=wmin
    return scale, zp


def init_mse_grid(w: torch.Tensor, group_size: int, ratios=(0.85, 0.9, 0.95, 1.0)) -> tuple[torch.Tensor, torch.Tensor]:
    """per-group MSE 网格搜索截断比例初始化（比 min-max 更稳）。"""
    wg = _reshape_groups(w, group_size)
    best_scale = None
    best_zp = None
    best_err = None
    for r in ratios:
        wmin = wg.amin(dim=-1)
        wmax = wg.amax(dim=-1)
        span = (wmax - wmin) * r
        center = (wmax + wmin) / 2
        lo, hi = center - span / 2, center + span / 2
        lo = torch.maximum(lo, wmin)
        hi = torch.minimum(hi, wmax)
        hi = torch.maximum(hi, lo + 1e-8)
        scale = (hi - lo) / (QMAX - QMIN)
        zp = -lo / scale
        q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + zp.unsqueeze(-1)), QMIN, QMAX)
        err = ((scale.unsqueeze(-1) * (q - zp.unsqueeze(-1)) - wg) ** 2).sum(dim=-1)
        if best_err is None:
            best_scale, best_zp, best_err = scale, zp, err
        else:
            better = err < best_err
            best_scale = torch.where(better, scale, best_scale)
            best_zp = torch.where(better, zp, best_zp)
            best_err = torch.where(better, err, best_err)
    return best_scale, best_zp


def quantize_weight(w: torch.Tensor, group_size: int, scale: torch.Tensor,
                    zp: torch.Tensor) -> GroupQuantResult:
    """按给定 scale/zp 量化权重，w (out,in)，scale/zp (out, in/g)。"""
    wg = _reshape_groups(w, group_size)
    q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + zp.unsqueeze(-1)), QMIN, QMAX)
    return GroupQuantResult(q=q.to(torch.int8), scale=scale, zp=zp)


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor,
                      zp: torch.Tensor) -> torch.Tensor:
    """NPU 真实反量化路径：scale * (q - zp)，还原 (out, in)。q 为分组形状 (out, in/g, g)。"""
    out_dim, n_groups, gs = q.shape
    w = scale.unsqueeze(-1) * (q.float() - zp.unsqueeze(-1))
    return w.reshape(out_dim, n_groups * gs)


def fake_quant_ste(w: torch.Tensor, group_size: int, scale: torch.Tensor,
                   zp: torch.Tensor) -> torch.Tensor:
    """校准期前向：量化-反量化，STE 使梯度绕过 round/clamp。"""
    wg = _reshape_groups(w, group_size)
    q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + zp.unsqueeze(-1)), QMIN, QMAX)
    dq = scale.unsqueeze(-1) * (q - zp.unsqueeze(-1))
    return w + (dq.reshape_as(w) - w).detach()  # STE
