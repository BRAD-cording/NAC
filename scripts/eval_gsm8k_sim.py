"""GSM8K 模拟量化精度评测（P1：NAC/GPTQ 任务级证据，数学推理面）。

板上 GSM8K 协议的 CPU sim 版（与 src/eval/accuracy.py eval_task 对齐）：
- 数据：openai/gsm8k "main" test，seed=42 随机 200 题（与板上同题集）
- prompt：GSM8K_FEWSHOT（2-shot CoT 前缀）+ "Question: {q}\nAnswer:"
- 生成式：512 tok greedy + extract_number（取最后一个数字），|pred-gold|<1e-4
- 配置：minmax / nac / 任意 label=qparams 文件，走 apply_qparams 同一数值路径

用法（板上 A72）：
  taskset -c 4-7 .venv/bin/python scripts/eval_gsm8k_sim.py \
      --model models/Qwen2.5-1.5B-Instruct-hf \
      --qparams results/nac_qwen1.5b_g64.qparams.pt --group-size 64 \
      --configs minmax,nac --out results/acc_gsm8k_sim_g64.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.accuracy import (GSM8K_FEWSHOT, GSM8K_TMPL, extract_answer_is,  # noqa: E402
                               extract_number, load_gsm8k)
from src.nac.awq import apply_fold, undo_fold_norms  # noqa: E402
from src.nac.validate_sim import apply_qparams, restore_from_disk  # noqa: E402

BACKUP = Path(__file__).resolve().parents[1] / "models" / "_gsm8k_sim_backup.pt"


def bootstrap_ci(ok, n_boot=1000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(ok)
    accs = sorted(sum(rng.choices(ok, k=n)) / n for _ in range(n_boot))
    return accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)]


@torch.no_grad()
def eval_config(model, tok, rows, bs, max_new, records, start, out_path, rec_path,
                stop_strings=None):
    n = len(rows)
    t0 = time.time()
    done = start
    while done < n:
        chunk = rows[done: done + bs]
        prompts = [GSM8K_FEWSHOT + GSM8K_TMPL.format(q=r["question"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True)
        input_len = enc.input_ids.shape[1]
        gen_kw = {}
        if stop_strings:
            # 2026-10-29：2-shot CoT 下所有方法都会在答完后续写下一题，
            # 末数字抽取因此读到续写文本里的数字（见 draft §6.2）。停止串让"算错/不停"分离。
            gen_kw = {"stop_strings": list(stop_strings), "tokenizer": tok}
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id, **gen_kw)
        texts = tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
        for j, row in enumerate(chunk):
            text = texts[j].strip()
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            pred = extract_number(text)
            ok = pred is not None and abs(pred - float(gold)) < 1e-4
            # dual-metric（2026-10-29 m1 复核处置 A 线）：answer-is 优先为第二指标，
            # 主指标 ok 仍用 extract_number 保持与历史口径可比
            pred_ai = extract_answer_is(text)
            ok_ai = pred_ai is not None and abs(pred_ai - float(gold)) < 1e-4
            records.append({"i": done + j, "pred": str(pred), "gold": gold,
                            "ok": ok, "pred_answeris": str(pred_ai),
                            "ok_answeris": ok_ai, "gen": text})
            done += 1
        accs = [r["ok"] for r in records]
        accs_ai = [r["ok_answeris"] for r in records]
        snapshot = {"num_evaluated": done, "acc": round(sum(accs) / done, 4),
                    "acc_answeris": round(sum(accs_ai) / done, 4),
                    "none_rate": round(sum(r["pred"] == "None" for r in records) / done, 4),
                    "elapsed_s": round(time.time() - t0, 1),
                    "sec_per_q": round((time.time() - t0) / done, 2)}
        out_path.write_text(json.dumps(snapshot, indent=2))
        rec_path.write_text(json.dumps(records, ensure_ascii=False))
        print(f"[sim-gsm8k] {done}/{n} acc={snapshot['acc']:.4f} "
              f"({snapshot['sec_per_q']}s/q, ETA {(n - done) * snapshot['sec_per_q'] / 3600:.1f}h)",
              flush=True)

    accs = [r["ok"] for r in records]
    accs_ai = [r["ok_answeris"] for r in records]
    lo, hi = bootstrap_ci(accs)
    return {"num_evaluated": done, "acc": round(sum(accs) / done, 4),
            "acc_ci95": [round(lo, 4), round(hi, 4)],
            "acc_answeris": round(sum(accs_ai) / done, 4),
            "none_rate": round(sum(r["pred"] == "None" for r in records) / done, 4),
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description="GSM8K simulated-quant accuracy eval (CPU)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--qparams", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--configs", default="minmax,nac")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="调试：只评前 N 题")
    ap.add_argument("--fold", default=None,
                    help="AWQ 折叠参数 .fold.pt（awq_calibrate.py 产物）；qparams 配置评测前折叠，评测后撤销 norms")
    ap.add_argument("--stop-strings", default="",
                    help="逗号分隔的停止串（如 '\\nQuestion:'）；空=不启用，保持与历史结果可比")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    print(f"[sim-gsm8k] loading {args.model} (fp32) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    rows = load_gsm8k(args.num_samples)  # seed=42，与板上同题集
    if args.limit:
        rows = rows[: args.limit]
    print(f"[sim-gsm8k] {len(rows)} questions (board-protocol, 2-shot CoT) "
          f"gs={args.group_size} bs={args.batch_size}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model, "group_size": args.group_size,
               "num_samples": args.num_samples, "max_new_tokens": args.max_new_tokens,
               "protocol": "2-shot CoT generative(512tok), batched sim fp32",
               "stop_strings": args.stop_strings or None,
               "configs": {}}
    stops = ([s.encode("utf-8").decode("unicode_escape")
              for s in args.stop_strings.split(",") if s] if args.stop_strings else None)
    fold = None
    if args.fold:
        fold = torch.load(args.fold, map_location="cpu", weights_only=True)
        print(f"[sim-gsm8k] fold params loaded from {args.fold} ({len(fold)} blocks)", flush=True)

    for spec in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if "=" in spec:
            cfg, qp_file = spec.split("=", 1)
        else:
            cfg, qp_file = spec, args.qparams
        print(f"[sim-gsm8k] ===== config: {cfg} =====", flush=True)
        rec_path = out_path.with_suffix(f".records.{cfg}.json")
        records = []
        if rec_path.exists():
            records = json.loads(rec_path.read_text())
            print(f"[sim-gsm8k] resume {cfg} from #{len(records)}", flush=True)
        if cfg == "fp16":
            pass
        elif cfg == "minmax":
            apply_qparams(model, args.group_size, "minmax", backup_path=str(BACKUP))
        else:
            if fold is not None:
                apply_fold(model, fold)  # AWQ：先折叠生产者/消费者权重再按 qparams 伪量化
            qparams = torch.load(qp_file, map_location="cpu", weights_only=True)
            apply_qparams(model, args.group_size, "nac", qparams, backup_path=str(BACKUP))
        try:
            stats = eval_config(model, tok, rows, args.batch_size,
                                args.max_new_tokens, records, len(records),
                                out_path, rec_path, stop_strings=stops)
        finally:
            if cfg != "fp16" and BACKUP.exists():
                restore_from_disk(model, str(BACKUP))
            if fold is not None and cfg not in ("fp16", "minmax"):
                undo_fold_norms(model, fold)  # linears 已由 restore_from_disk 精确还原
        results["configs"][cfg] = stats
        print(f"[sim-gsm8k] {cfg}: {json.dumps(stats)}", flush=True)
        out_path.with_suffix(".full.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False))

    # 配对差：nac vs minmax（McNemar 不一致对）
    def load_ok(cfg):
        p = out_path.with_suffix(f".records.{cfg}.json")
        if not p.exists():
            return None
        return [r["ok"] for r in json.loads(p.read_text())]

    if "minmax" in args.configs and "nac" in args.configs:
        mm, nc = load_ok("minmax"), load_ok("nac")
        if mm and nc:
            n = min(len(mm), len(nc))
            b = sum(1 for a, c in zip(mm[:n], nc[:n]) if a and not c)
            c_ = sum(1 for a, c in zip(mm[:n], nc[:n]) if not a and c)
            results["paired_nac_vs_minmax"] = {
                "n": n, "acc_diff_pp": round((sum(nc[:n]) - sum(mm[:n])) / n * 100, 2),
                "mcnemar_b_minmax_only": b, "mcnemar_c_nac_only": c_}
            out_path.with_suffix(".full.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False))
            print(f"[sim-gsm8k] paired nac-minmax: {results['paired_nac_vs_minmax']}",
                  flush=True)
    print(f"[sim-gsm8k] -> {out_path.with_suffix('.full.json')}", flush=True)


if __name__ == "__main__":
    main()
