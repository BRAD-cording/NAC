"""Prefill 阈值扫描（贡献2，方案 D12–D13）。

实测 RK3576 上 prefill 延迟随序列长度变化：
- 短序列 NPU prefill 快（compute-bound 区间）
- 长序列 CPU（llama.cpp）反而更快（内存带宽瓶颈区间的量化 dequant 开销）

扫描 [128..4096] 各点，对 CPU/NPU 两条曲线拟合交叉点 L*，输出 results/route_threshold.json。
CPU 侧用 llama.cpp（llama-cli）；未安装时仅采 NPU 曲线并提示。
"""
import argparse
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from ..runtime.llm_client import RKLLMClient
from ..utils.freq import current_freqs


def npu_prefill_curve(client: RKLLMClient, tokenizer, lengths, repeats) -> list[dict]:
    """NPU 侧 prefill 延迟：短生成（1 token 级）下 total_ms ≈ prefill + 常数。"""
    points = []
    for n in lengths:
        prompt = _synth_prompt(tokenizer, n)
        ttfts = []
        for _ in range(repeats):
            r = client.generate(prompt)
            ttfts.append(r["ttft_ms"])
        points.append({"backend": "npu", "len": n,
                       "prefill_ms": round(statistics.median(ttfts), 1)})
        print(f"[scan] npu len={n:5d} prefill={points[-1]['prefill_ms']:8.1f}ms")
    return points


def cpu_prefill_curve(llama_cli: str, model_gguf: str, tokenizer, lengths,
                      repeats: int, threads: int = 6) -> list[dict]:
    """CPU 侧 prefill：llama-cli -p <prompt> -n 1，解析 timing 输出。"""
    points = []
    for n in lengths:
        prompt = _synth_prompt(tokenizer, n)
        pre = []
        for _ in range(repeats):
            proc = subprocess.run(
                [llama_cli, "-m", model_gguf, "-p", prompt, "-n", "1",
                 "-t", str(threads), "--no-display-prompt"],
                capture_output=True, text=True, timeout=600)
            pre.append(_parse_llama_prefill(proc.stderr + proc.stdout))
        pre = [p for p in pre if p]
        if not pre:
            print(f"[scan] cpu len={n}: 解析失败，检查 llama-cli 输出")
            continue
        points.append({"backend": "cpu", "len": n,
                       "prefill_ms": round(statistics.median(pre), 1)})
        print(f"[scan] cpu len={n:5d} prefill={points[-1]['prefill_ms']:8.1f}ms")
    return points


def cpu_prefill_curve_lcpp(gguf: str, tokenizer, lengths, repeats: int,
                           threads: int = 4, n_ctx: int = 4096) -> list[dict]:
    """CPU 侧 prefill：llama-cpp-python（板上无 llama-cli 时的替代实现）。

    tokenize 后对 llm.eval(ids) 计墙钟时间 = 纯 prefill（KV 已 reset）。
    """
    import time as _time

    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_threads=threads, n_ctx=n_ctx,
                n_batch=n_ctx, verbose=False, logits_all=False)
    points = []
    for n in lengths:
        prompt = _synth_prompt(tokenizer, n)
        pre = []
        for _ in range(repeats):
            llm.reset()
            ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)
            t0 = _time.perf_counter()
            llm.eval(ids)
            pre.append((_time.perf_counter() - t0) * 1000)
        points.append({"backend": "cpu", "len": len(ids),
                       "prefill_ms": round(statistics.median(pre), 1)})
        print(f"[scan] cpu(lcpp) len={len(ids):5d} prefill={points[-1]['prefill_ms']:8.1f}ms")
    return points


def _parse_llama_prefill(text: str) -> float | None:
    """llama-cli 输出形如 'prompt eval time =  1234.56 ms /   512 tokens'。"""
    import re
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", text)
    return float(m.group(1)) if m else None


def _synth_prompt(tokenizer, target_len: int) -> str:
    filler = ("The quick brown fox jumps over the lazy dog. "
              "Edge inference on memory-constrained SoCs depends on "
              "weight quantization and bandwidth scheduling. ")
    ids = tokenizer(filler).input_ids
    reps = target_len // len(ids) + 1
    return tokenizer.decode((ids * reps)[:target_len])


def find_threshold(points: list[dict]) -> int | None:
    """两条曲线交叉点：L* = argmin{len : cpu prefill < npu prefill}。"""
    cpu = {p["len"]: p["prefill_ms"] for p in points if p["backend"] == "cpu"}
    npu = {p["len"]: p["prefill_ms"] for p in points if p["backend"] == "npu"}
    common = sorted(set(cpu) & set(npu))
    for n in common:
        if cpu[n] < npu[n]:
            return n
    return None


def main():
    ap = argparse.ArgumentParser(description="Prefill routing threshold scan")
    ap.add_argument("--model", required=True, help=".rkllm")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--llama-cli", default=None, help="llama-cli 路径（可选）")
    ap.add_argument("--gguf", default=None, help="CPU 侧 gguf 模型")
    ap.add_argument("--lcpp-gguf", default=None,
                    help="llama-cpp-python 用的 gguf（板上无 llama-cli 时）")
    ap.add_argument("--lcpp-threads", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 1536, 2048, 3072, 4096])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out_path = Path(args.out or f"results/route_threshold_{datetime.now():%Y%m%d_%H%M%S}.json")
    points = []

    with RKLLMClient(args.model, max_new_tokens=8) as client:
        client.generate("Hello")
        points += npu_prefill_curve(client, tok, args.lengths, args.repeats)

    if args.llama_cli and args.gguf:
        points += cpu_prefill_curve(args.llama_cli, args.gguf, tok,
                                    args.lengths, args.repeats)
    elif args.lcpp_gguf:
        points += cpu_prefill_curve_lcpp(args.lcpp_gguf, tok, args.lengths,
                                         args.repeats, threads=args.lcpp_threads)
    else:
        print("[scan] 未提供 llama-cli/gguf 或 lcpp-gguf，仅采集 NPU 曲线")

    l_star = find_threshold(points)
    results = {"model": args.model, "freqs": current_freqs(), "L_star": l_star,
               "points": points}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[scan] L* = {l_star} -> {out_path}")


if __name__ == "__main__":
    main()
