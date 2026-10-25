"""NAC 验证（T5 / 方案贡献1证据）：CPU 模拟量化 PPL 对比。

三种配置在同一 held-out wikitext-2 test 序列上比较困惑度：
  fp32    : 未量化参考（float32 权重与计算；板上 bf16 走 Aten 朴素核慢 3×，
            fp32 与量化路径反量化数值一致，三方同基准可比）
  minmax  : per-group min-max scale/zp（RTN 基线，等价 toolkit 默认 normal 算法）
  nac     : calibrate.py 学习到的 scale/zp（results/*.qparams.pt）

说明：.rkllm 真实导出需 x86 toolkit（板上不可用），本脚本以 torch 模拟
Rockchip W4A16 反量化公式 W = scale*(q - zp) 在权重上做伪量化，
数值上等价于 NPU 反量化路径（见 paper Method 节）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nac.quantize import dequantize_weight, init_minmax, quantize_weight  # noqa: E402


def load_heldout(tokenizer, n_seqs: int, seqlen: int):
    """wikitext-2 test 集（校准用的是 validation，互不重叠）。"""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seqlen
    idx = torch.linspace(0, n - 1, n_seqs).long()  # 均匀取段
    return torch.stack([ids[i * seqlen: (i + 1) * seqlen] for i in idx])


@torch.no_grad()
def eval_ppl(model, seqs: torch.Tensor, bs: int = 1) -> float:
    nll, n_tok = 0.0, 0
    for i in range(0, seqs.shape[0], bs):
        chunk = seqs[i: i + bs]
        out = model(chunk[:, :-1])
        tgt = chunk[:, 1:]
        nll += torch.nn.functional.cross_entropy(
            out.logits.flatten(0, 1).float(), tgt.flatten(), reduction="sum").item()
        n_tok += tgt.numel()
        print(f"    [ppl] {min(i + bs, seqs.shape[0])}/{seqs.shape[0]} seqs", flush=True)
    return float(torch.exp(torch.tensor(nll / n_tok)))


def linears(block: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    return {n: m for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)}


@torch.no_grad()
def apply_qparams(model, gs: int, mode: str, qparams=None, backup_path=None):
    """对模型所有 decoder block 的 Linear 做伪量化写回。

    mode=minmax: 由 fp 权重计算 min-max scale/zp
    mode=nac:    使用 qparams[block_idx][name] 的 scale/zp
    backup_path 给定时把 fp 权重存盘（零克隆，防 RAM OOM），返回 None；
    否则返回 {block_idx: {name: weight_backup}}（RAM 备份）。
    """
    if backup_path:
        payload = {}
        for bi, block in enumerate(model.model.layers):
            payload[bi] = {n: m.weight.data.to(torch.bfloat16) for n, m in linears(block).items()}
        torch.save(payload, backup_path)
        payload = None
    backup: dict[int, dict] = {}
    for bi, block in enumerate(model.model.layers):
        backup[bi] = {}
        for name, mod in linears(block).items():
            if not backup_path:
                backup[bi][name] = mod.weight.data.clone()
            if mode == "minmax":
                scale, zp = init_minmax(mod.weight.data.float(), gs)
            else:
                qp = qparams[bi][name]
                scale, zp = qp["scale"], qp["zp"]
            res = quantize_weight(mod.weight.data.float(), gs, scale, zp)
            mod.weight.data = dequantize_weight(res.q, res.scale, res.zp).to(mod.weight.dtype)
    return backup


@torch.no_grad()
def restore(model, backup: dict):
    for bi, block in enumerate(model.model.layers):
        for name, mod in linears(block).items():
            mod.weight.data = backup[bi][name]


@torch.no_grad()
def restore_from_disk(model, backup_path):
    payload = torch.load(backup_path, map_location="cpu", weights_only=True)
    for bi, block in enumerate(model.model.layers):
        mods = dict(linears(block))
        for name, mod in mods.items():
            mod.weight.data.copy_(payload[bi][name].to(mod.weight.dtype))
    Path(backup_path).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="NAC simulated-quant validation (PPL)")
    ap.add_argument("--model", required=True, help="HF 模型目录（FP16 权重）")
    ap.add_argument("--qparams", required=True, help="calibrate.py 输出的 .qparams.pt")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--num-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--fp-ppl", type=float, default=None,
                    help="复用已测 fp 参考 PPL（fp 参考与 group-size 无关，g64 轮可复用 g128 值省 1/3 时长）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    print(f"[val] loading {args.model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    gs = args.group_size

    seqs = load_heldout(tok, args.num_seqs, args.seqlen)
    print(f"[val] held-out: {seqs.shape[0]} seqs x {seqs.shape[1]} tokens")

    results = {"model": args.model, "group_size": gs, "num_seqs": int(seqs.shape[0]),
               "seqlen": args.seqlen, "ppl": {}}

    t0 = time.time()
    if args.fp_ppl is not None:
        ppl_fp = args.fp_ppl
        print(f"[val] fp16   PPL = {ppl_fp:.3f}  (reused)")
    else:
        ppl_fp = eval_ppl(model, seqs, bs=2)
        print(f"[val] fp16   PPL = {ppl_fp:.3f}  ({time.time() - t0:.0f}s)")
    results["ppl"]["fp16"] = round(ppl_fp, 3)

    bak = "/tmp/val_backup_minmax.pt"
    apply_qparams(model, gs, "minmax", backup_path=bak)
    t0 = time.time()
    ppl_mm = eval_ppl(model, seqs, bs=2)
    results["ppl"]["minmax"] = round(ppl_mm, 3)
    print(f"[val] minmax  PPL = {ppl_mm:.3f}  ({time.time() - t0:.0f}s)")
    restore_from_disk(model, bak)

    qparams = torch.load(args.qparams, map_location="cpu", weights_only=True)
    bak = "/tmp/val_backup_nac.pt"
    apply_qparams(model, gs, "nac", qparams, backup_path=bak)
    t0 = time.time()
    ppl_nac = eval_ppl(model, seqs, bs=2)
    results["ppl"]["nac"] = round(ppl_nac, 3)
    print(f"[val] nac     PPL = {ppl_nac:.3f}  ({time.time() - t0:.0f}s)")
    restore_from_disk(model, bak)

    results["deltas"] = {
        "minmax_vs_fp": round(ppl_mm - ppl_fp, 3),
        "nac_vs_fp": round(ppl_nac - ppl_fp, 3),
        "nac_improvement_pct": round((ppl_mm - ppl_nac) / ppl_mm * 100, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[val] -> {out}\n{json.dumps(results['ppl'], indent=1)}")


if __name__ == "__main__":
    main()
