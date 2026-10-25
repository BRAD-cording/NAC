"""NAC-v2 校准（问题 A 修复：Hessian 补偿 + 激活加权 + integer-zp）。

与 GPTQ 基线完全同构的公平对比框架（同一校准数据 / 块级顺序传播 / qparams
格式 {block: {name: {scale, zp}}}），差异仅在组参数搜索核：
- GPTQ 基线：均匀加权 maxshrink 网格（100 点），zp 取整
- NAC-v2：  激活加权（diag(H)）κ 截断网格（9 点），int-zp 可选（默认开）

用法（板上 A72，须在内存窗口内运行，勿与 sim 评测/构建并行）：
  taskset -c 4-7 .venv/bin/python scripts/nac_v2_calibrate.py \
      --model models/Qwen2.5-1.5B-Instruct-hf --group-size 64 \
      --out results/nacv2_qwen1.5b_g64
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nac.hessian import DEFAULT_RATIOS, gptq_aw_linear  # noqa: E402
from src.nac.quantize import QMAX, QMIN  # noqa: E402
from src.utils.data import load_calib_samples  # noqa: E402

LINEAR_TYPES = (torch.nn.Linear,)


def main():
    ap = argparse.ArgumentParser(description="NAC-v2 calibration (CPU)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "nac.yaml"))
    ap.add_argument("--out", required=True, help="输出前缀，如 results/nacv2_qwen1.5b_g64")
    ap.add_argument("--float-zp", action="store_true",
                    help="关闭 integer-zp 约束（消融用；默认整格点，同服问题 B）")
    ap.add_argument("--limit-blocks", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["nac"]
    gs = args.group_size
    int_zp = not args.float_zp
    torch.set_num_threads(4)
    print(f"[nacv2] loading {args.model} (fp32) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    samples = load_calib_samples(cfg["calib_dataset"], cfg["num_samples"], tokenizer, cfg["seqlen"])
    input_ids = torch.cat(samples, dim=0)[: cfg["capture_samples"]]
    print(f"[nacv2] calib seqs={input_ids.shape[0]} seqlen={cfg['seqlen']} "
          f"group={gs} int_zp={int_zp}", flush=True)

    blocks = model.model.layers
    n_blocks = len(blocks) if args.limit_blocks is None else args.limit_blocks

    # 1) FP 前向捕获块输入（顺序传播用）
    block_inputs: list[torch.Tensor] = []
    hooks = []

    def block_pre_hook(_idx):
        def hook(_mod, aargs, _kw):
            block_inputs.append(aargs[0].detach().float().cpu())
        return hook

    for i, b in enumerate(blocks[:n_blocks]):
        hooks.append(b.register_forward_pre_hook(block_pre_hook(i), with_kwargs=True))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
    print(f"[nacv2] captured {len(block_inputs)} block inputs", flush=True)

    # 2) 逐块：块前向捕获 H → v2 核量化写回 → 量化传播
    qparams: dict[int, dict] = {}
    t0 = time.time()
    pbar = tqdm(range(n_blocks), desc="NAC-v2 blocks")
    for bi in pbar:
        x = block_inputs[bi]
        hess: dict[str, torch.Tensor] = {}
        tok_counts: dict[str, int] = {}
        hooks = []

        def linear_pre_hook(name):
            def hook(_mod, aargs, _kw):
                xx = aargs[0].detach().reshape(-1, aargs[0].shape[-1]).float()
                h = hess.get(name)
                if h is None:
                    h = xx.shape[1]
                    hess[name] = torch.zeros(h, h)
                    tok_counts[name] = 0
                hess[name] += 2.0 * (xx.T @ xx)
                tok_counts[name] += xx.shape[0]
            return hook

        for name, mod in blocks[bi].named_modules():
            if isinstance(mod, LINEAR_TYPES):
                hooks.append(mod.register_forward_pre_hook(linear_pre_hook(name),
                                                           with_kwargs=True))
        with torch.no_grad():
            pos_ids = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
            pos_emb = model.model.rotary_emb(x, pos_ids)
            blocks[bi](x, position_embeddings=pos_emb)
        for h in hooks:
            h.remove()

        qp_i: dict[str, dict] = {}
        for name, mod in blocks[bi].named_modules():
            if not isinstance(mod, LINEAR_TYPES):
                continue
            H = hess[name] / tok_counts[name]  # 平均 Hessian（尺度不影响解）
            w = mod.weight.data.float()
            aw = torch.diagonal(H).clone()  # 通道激活权重 = H 对角（二阶矩）
            scale, zp = gptq_aw_linear(w, H, gs, aw=aw, int_zp=int_zp)
            mod.weight.data = (scale.unsqueeze(-1) * (
                torch.clamp(torch.round(
                    w.reshape(w.shape[0], -1, gs) / scale.unsqueeze(-1) + zp.unsqueeze(-1)),
                    QMIN, QMAX) - zp.unsqueeze(-1))).reshape(w.shape)
            qp_i[name] = {"scale": scale, "zp": zp}
        qparams[bi] = qp_i
        if bi + 1 < n_blocks:
            with torch.no_grad():
                block_inputs[bi + 1] = blocks[bi](
                    x, position_embeddings=pos_emb).detach().float().cpu()
        del x, hess
        pbar.set_postfix(sec=round(time.time() - t0, 0))

    # 3) 导出（与 NAC/GPTQ qparams 同格式，eval 侧 label=qparams_file 直接可用）
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qparams, str(out) + ".qparams.pt")
    meta = {"model": args.model, "method": "nac_v2", "group_size": gs,
            "bits": 4, "asym": True, "act_order": False,
            "int_zp": int_zp, "act_weight": "diag(H)",
            "ratios": list(DEFAULT_RATIOS),
            "sequential": "block (layer-parallel within block)",
            "percdamp": 0.01,
            "calib": {"dataset": cfg["calib_dataset"], "num_samples": cfg["num_samples"],
                      "capture_samples": int(input_ids.shape[0]), "seqlen": cfg["seqlen"]},
            "blocks_calibrated": n_blocks, "device": "cpu",
            "seconds": round(time.time() - t0, 1)}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[nacv2] done in {meta['seconds']}s -> {out}.qparams.pt", flush=True)


if __name__ == "__main__":
    main()
