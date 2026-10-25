"""NAC：NPU 感知 PTQ 校准（论文贡献1，方案 D8–D11）。

板上 CPU 预算下的可行实现（默认 --mode grid）：
GPTQ-style 顺序量化传播 + AWQ-style 激活加权 per-group 网格搜索：
1. FP 路径前向：pre-hook 捕获每个 decoder block 的输入 hidden（顺序传播用），
   linear pre-hook 捕获每个 Linear 的按通道激活幅值 |x| 均值（加权用）
2. 逐 block i（量化输出传播到下一 block）：
   a. 对每个 Linear：per-group 网格搜索截断比例 κ（组间独立取最优），
      最小化激活加权量化误差 Σ_g min_κ a ⊙ (W - deq(W; κ, g))²
      —— 纯权重张量运算（无需 block forward），与 NPU 反量化公式一致
   b. dequant 权重写回活体模型，块前向一次得到下一 block 的真实输入
3. 导出 .qparams.pt（每 Linear 的 per-group scale/zp）+ .meta.json

说明：
- CPU 上以 float32 计算（RK3576 的 A72/A53 无 bf16 硬件支持）
- Adam-STE 逐块可学习 scale/zp 版本保留在 --mode adam（x86+GPU 路线；
  板上实测 82 TFLOP/块 × 28 块 ≈ 87h 不可行，见 results/nac_g128.log）
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from ..utils.data import load_calib_samples
from .quantize import QMIN, QMAX  # noqa: F401  (uint4 域)

LINEAR_TYPES = (nn.Linear,)

DEFAULT_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4)


def grid_search_groups(w: torch.Tensor, act: torch.Tensor, group_size: int,
                       ratios=DEFAULT_RATIOS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """per-group 激活加权网格搜索截断比例。

    w: (out, in) fp32 权重；act: (in,) 按通道激活幅值（|x| 均值）。
    返回 (scale, zp, best_ratio) 形状 (out, in/g)；每组独立选最优 κ。
    """
    out_dim, in_dim = w.shape
    ng = in_dim // group_size
    wg = w.reshape(out_dim, ng, group_size)
    ag = act.reshape(ng, group_size)  # 组内各通道激活权重
    wmin, wmax = wg.amin(dim=-1), wg.amax(dim=-1)
    wmax = torch.maximum(wmax, wmin + 1e-8)

    best_scale = best_zp = best_err = best_r = None
    for r in ratios:
        span = (wmax - wmin) * r
        center = (wmax + wmin) / 2
        lo = torch.maximum(center - span / 2, wmin)
        hi = torch.minimum(center + span / 2, wmax)
        hi = torch.maximum(hi, lo + 1e-8)
        scale = (hi - lo) / (QMAX - QMIN)
        zp = -lo / scale
        q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + zp.unsqueeze(-1)), QMIN, QMAX)
        err = (((scale.unsqueeze(-1) * (q - zp.unsqueeze(-1)) - wg) ** 2) * ag).sum(dim=-1)
        if best_err is None:
            best_scale, best_zp, best_err, best_r = scale, zp, err, torch.full_like(scale, r)
        else:
            better = err < best_err
            best_scale = torch.where(better, scale, best_scale)
            best_zp = torch.where(better, zp, best_zp)
            best_err = torch.where(better, err, best_err)
            best_r = torch.where(better, torch.full_like(scale, r), best_r)
    return best_scale, best_zp, best_r


def dequant_from_qparams(w: torch.Tensor, group_size: int,
                          scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    """NPU 反量化路径：scale*(q-zp)，q 由当前 scale/zp 量化得到。"""
    out_dim, in_dim = w.shape
    wg = w.reshape(out_dim, -1, group_size)
    q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + zp.unsqueeze(-1)), QMIN, QMAX)
    return (scale.unsqueeze(-1) * (q - zp.unsqueeze(-1))).reshape(out_dim, in_dim)


def main():
    ap = argparse.ArgumentParser(description="NAC: NPU-aware PTQ calibration (CPU)")
    ap.add_argument("--model", required=True, help="HF hub id 或本地路径")
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "nac.yaml"))
    ap.add_argument("--out", required=True, help="输出前缀，如 results/nac_qwen2.5-1.5b_g128")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--mode", choices=["grid", "adam"], default="grid",
                    help="grid: 激活加权网格搜索（板上默认）；adam: STE 可学习（x86）")
    ap.add_argument("--limit-blocks", type=int, default=None, help="调试：只校准前 N 个 block")
    ap.add_argument("--save-weights", action="store_true",
                    help="同时保存反量化 state_dict（+6GB，x86 导出用；板上默认关）")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["nac"]
    if args.group_size:
        cfg["group_size"] = args.group_size
    if args.num_samples:
        cfg["num_samples"] = args.num_samples
    gs = cfg["group_size"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)

    print(f"[NAC] loading {args.model} (CPU) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # float32 加载：RK3576（A72/A53）无 bf16 硬件支持，bf16 会落入极慢的
    # mkldnn 回退路径；fp32 NEON BLAS 快一个量级（论文 Method：CPU float32 计算）
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    samples = load_calib_samples(cfg["calib_dataset"], cfg["num_samples"], tokenizer, cfg["seqlen"])
    input_ids = torch.cat(samples, dim=0)[: cfg["capture_samples"]]
    print(f"[NAC] calib seqs={input_ids.shape[0]} seqlen={cfg['seqlen']} group={gs} mode={args.mode}")

    blocks = model.model.layers
    n_blocks = len(blocks) if args.limit_blocks is None else args.limit_blocks

    # 1) FP 前向捕获：block 输入（顺序传播用）+ 每个 Linear 的通道激活幅值
    block_inputs: list[torch.Tensor] = []
    act_stats: list[dict[str, torch.Tensor]] = [dict() for _ in range(n_blocks)]
    hooks = []

    def block_pre_hook(idx):
        def hook(_mod, aargs, _kw):
            block_inputs.append(aargs[0].detach().float().cpu())
        return hook

    def linear_pre_hook(bi, name):
        def hook(_mod, aargs, _kw):
            x = aargs[0].detach()
            act_stats[bi][name] = x.abs().reshape(-1, x.shape[-1]).mean(0).half().cpu()
        return hook

    for i, b in enumerate(blocks[:n_blocks]):
        hooks.append(b.register_forward_pre_hook(block_pre_hook(i), with_kwargs=True))
        for name, mod in b.named_modules():
            if isinstance(mod, LINEAR_TYPES):
                hooks.append(mod.register_forward_pre_hook(linear_pre_hook(i, name),
                                                           with_kwargs=True))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
    print(f"[NAC] captured {len(block_inputs)} block inputs, "
          f"{sum(len(s) for s in act_stats)} linear act stats")

    # 2) 逐 block 网格搜索 + 写回 + 量化传播
    t0 = time.time()
    qparams: dict[int, dict] = {}
    pbar = tqdm(range(n_blocks), desc="NAC blocks")
    for i in pbar:
        x = block_inputs[i]
        qp_i: dict[str, dict] = {}
        for name, mod in blocks[i].named_modules():
            if not isinstance(mod, LINEAR_TYPES):
                continue
            w = mod.weight.data.float()
            act = act_stats[i][name].float()
            scale, zp, ratio = grid_search_groups(w, act, gs)
            mod.weight.data = dequant_from_qparams(w, gs, scale, zp).to(mod.weight.dtype)
            qp_i[name] = {"scale": scale, "zp": zp, "ratio": ratio}
        qparams[i] = qp_i
        # 量化路径传播（GPTQ-style sequential）
        # 直接调用 decoder block 需显式传 position_embeddings（新版 transformers
        # 由顶层模型计算 rotary 后下传），否则内部 cos, sin = None 崩溃
        if i + 1 < n_blocks:
            with torch.no_grad():
                pos_ids = torch.arange(x.shape[1], device=x.device)
                pos_ids = pos_ids.unsqueeze(0).expand(x.shape[0], -1)
                pos_emb = model.model.rotary_emb(x, pos_ids)
                block_inputs[i + 1] = blocks[i](
                    x, position_embeddings=pos_emb).detach().float().cpu()
        pbar.set_postfix(ratio_mean=round(
            float(torch.cat([v["ratio"].flatten() for v in qp_i.values()]).mean()), 3))
        del x

    # 3) 导出
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_weights:
        torch.save(model.state_dict(), str(out) + ".weights.pt")
    torch.save(qparams, str(out) + ".qparams.pt")
    ratio_mean = float(torch.cat(
        [v["ratio"].flatten() for qp in qparams.values() for v in qp.values()]).mean())
    meta = {
        "model": args.model, "group_size": gs, "bits": cfg["bits"],
        "mode": args.mode, "ratios": list(DEFAULT_RATIOS), "ratio_mean": round(ratio_mean, 4),
        "blocks_calibrated": n_blocks,
        "num_samples": cfg["num_samples"], "capture_samples": int(input_ids.shape[0]),
        "seqlen": cfg["seqlen"], "propagate": "quant", "device": "cpu",
        "qparams_file": str(out) + ".qparams.pt",
        "seconds": round(time.time() - t0, 1),
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[NAC] done in {meta['seconds']}s -> {out}.qparams.pt / .meta.json")


if __name__ == "__main__":
    main()
