"""板上性能评测：TTFT / decode tok/s / 端到端延迟（方案 3.3、D5）。

扫描 prompt 长度 128/512/1024/2048，每点 N 次重复取中位数，
可选同时采样功耗（power.py）得到能效。输出 results/perf_*.json。
"""
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from ..runtime.llm_client import RKLLMClient
from ..utils.freq import current_freqs
from .power import PowerSampler

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "Edge large language model inference requires careful "
          "quantization of weights and activations for memory "
          "efficiency on resource constrained devices. ")


def build_prompt(tokenizer, target_len: int) -> str:
    """重复填充语料并按 token 数截断，构造约 target_len 个 token 的 prompt。"""
    ids = tokenizer(FILLER).input_ids
    reps = target_len // len(ids) + 2
    chunk = (ids * reps)[:target_len]
    return tokenizer.decode(chunk)


def run_point(client: RKLLMClient, prompt: str, repeats: int,
              sampler: PowerSampler | None) -> list[dict]:
    rows = []
    for _ in range(repeats):
        if sampler:
            sampler.start()
        r = client.generate(prompt)
        if sampler:
            power_w, energy_j = sampler.stop(results_tokens=r["n_tokens"])
            r["power_w"], r["energy_j"] = power_w, energy_j
            r["voltage_v"] = sampler.last_voltage_v
        rows.append(r)
    return rows


def aggregate(rows: list[dict]) -> dict:
    def med(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.median(vals), 2) if vals else None

    agg = {"ttft_ms_median": med("ttft_ms"),
           "decode_tps_median": med("decode_tps"),
           "total_ms_median": med("total_ms"),
           "n_tokens_median": med("n_tokens")}
    pw = [r["power_w"] for r in rows if r.get("power_w")]
    ej = [r["energy_j"] for r in rows if r.get("energy_j")]
    vv = [r["voltage_v"] for r in rows if r.get("voltage_v")]
    if vv:
        agg["voltage_v_mean"] = round(statistics.mean(vv), 3)
    if pw:
        agg["power_w_mean"] = round(statistics.mean(pw), 3)
    if ej:
        agg["energy_j_mean"] = round(statistics.mean(ej), 3)
        agg["tokens_per_j"] = round(agg["n_tokens_median"] / statistics.mean(ej), 2) \
            if agg["n_tokens_median"] else None
    return agg


def main():
    ap = argparse.ArgumentParser(description="On-device performance benchmark")
    ap.add_argument("--model", required=True, help=".rkllm 路径")
    ap.add_argument("--tokenizer", default="models/tokenizer", help="tokenizer 目录")
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--power", action="store_true", help="同时采样功耗")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lib", default=None, help="librkllmrt.so 路径（默认 1.2.3）")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    stem = Path(args.model).stem
    out_path = Path(args.out or f"results/perf_{stem}_{datetime.now():%Y%m%d_%H%M%S}.json")
    sampler = PowerSampler(interval=0.2) if args.power else None

    results = {"model": args.model, "max_new_tokens": args.max_new_tokens,
               "repeats": args.repeats, "freqs": current_freqs(), "points": []}
    with RKLLMClient(args.model, lib_path=args.lib,
                     max_new_tokens=args.max_new_tokens) as client:
        client.generate("Hello")  # warmup
        for plen in args.prompt_len:
            prompt = build_prompt(tok, plen)
            actual = len(tok(prompt).input_ids)
            rows = run_point(client, prompt, args.repeats, sampler)
            agg = aggregate(rows)
            agg.update({"prompt_len_target": plen, "prompt_len_actual": actual})
            results["points"].append(agg)
            pw = f" power={agg.get('power_w_mean')}W" if args.power else ""
            print(f"[perf] len={actual:5d} ttft={agg['ttft_ms_median']:8.1f}ms "
                  f"decode={agg['decode_tps_median']:6.2f}tok/s "
                  f"total={agg['total_ms_median']:8.1f}ms{pw}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[perf] -> {out_path}")


if __name__ == "__main__":
    main()
