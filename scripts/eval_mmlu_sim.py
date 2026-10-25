"""MMLU 模拟量化精度评测（论文 P0-2：NAC 任务级证据）。

板上 NPU 评测协议的 CPU sim 版（协议逐项对齐，见 src/eval/accuracy.py）：
- 数据：cais/mmlu "all" test，seed=42 分层子采样（与板上 acc_mmlu_*.json 同一题集）
- prompt：zero-shot（fewshot_num=0，"Answer with a single letter" 指令）
- 双协议：生成式（32 tok greedy + 正则抽取）+ 首 token 字母匹配，
  单次 generate 同时产出，协议损耗可直接对比
- 配置：minmax（RTN 基线）/ nac（qparams 伪量化）/ fp16（可选参考），
  伪量化走 src.nac.validate_sim.apply_qparams，与 PPL 验证同一数值路径
  （W = scale*(q-zp)，等价 NPU 反量化）

性能：批量左 padding generate（KV cache），taskset A72（4-7 核）；
逐批增量落盘（JSON + records），中断可续（--skip N 跳过已评题数）。

用法（板上）：
  taskset -c 4-7 .venv/bin/python scripts/eval_mmlu_sim.py \
      --model models/Qwen2.5-1.5B-Instruct-hf \
      --qparams results/nac_qwen1.5b_g64.qparams.pt --group-size 64 \
      --configs minmax,nac --out results/acc_mmlu_sim_g64.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.accuracy import LETTERS, extract_choice, load_mmlu, mmlu_prompt  # noqa: E402
from src.nac.awq import apply_fold, undo_fold_norms  # noqa: E402
from src.nac.validate_sim import apply_qparams, restore_from_disk  # noqa: E402

FIRSTTOK_MAP = {32: "A", 362: "A", 33: "B", 425: "B",
                34: "C", 356: "C", 35: "D", 422: "D"}
BACKUP = Path(__file__).resolve().parents[1] / "models" / "_mmlu_sim_backup.pt"


def bootstrap_ci(correct: list[bool], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """准确率的 percentile bootstrap 95% CI。"""
    rng = random.Random(seed)
    n = len(correct)
    accs = sorted(sum(rng.choices(correct, k=n)) / n for _ in range(n_boot))
    return accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)]


@torch.no_grad()
def eval_config(model, tok, rows, bs: int, max_new: int, records: list,
                start: int, out_path: Path, rec_path: Path) -> dict:
    """批量生成式评测，逐批增量落盘。records 就地追加（续跑时已有 start 条）。"""
    n = len(rows)
    t0 = time.time()
    done = start
    while done < n:
        chunk = rows[done: done + bs]
        prompts = [mmlu_prompt(r, fewshot_num=0) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True)
        input_len = enc.input_ids.shape[1]
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen_ids = out[:, input_len:]
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
        for j, row in enumerate(chunk):
            text = texts[j].strip()
            pred = extract_choice(text)
            ftok = FIRSTTOK_MAP.get(int(gen_ids[j, 0]))
            gold = LETTERS[row["answer"]]
            records.append({"i": done + j, "pred_gen": pred, "pred_ftok": ftok,
                            "gold": gold, "gen": text[:200],
                            "gen_ids_head": gen_ids[j, :4].tolist()})
            done += 1
        accs_gen = [r["pred_gen"] == r["gold"] for r in records]
        accs_ft = [r["pred_ftok"] == r["gold"] for r in records]
        snapshot = {
            "num_evaluated": done, "acc_gen": round(sum(accs_gen) / done, 4),
            "acc_ftok": round(sum(accs_ft) / done, 4),
            "none_gen_rate": round(sum(r["pred_gen"] is None for r in records) / done, 4),
            "elapsed_s": round(time.time() - t0, 1),
            "sec_per_q": round((time.time() - t0) / done, 2),
        }
        out_path.write_text(json.dumps(snapshot, indent=2))
        rec_path.write_text(json.dumps(records, ensure_ascii=False))
        eta = (n - done) * snapshot["sec_per_q"] / 3600
        print(f"[sim-mmlu] {done}/{n} acc_gen={snapshot['acc_gen']:.4f} "
              f"acc_ftok={snapshot['acc_ftok']:.4f} "
              f"({snapshot['sec_per_q']}s/q, ETA {eta:.1f}h)", flush=True)

    accs_gen = [r["pred_gen"] == r["gold"] for r in records]
    accs_ft = [r["pred_ftok"] == r["gold"] for r in records]
    lo_g, hi_g = bootstrap_ci(accs_gen)
    lo_f, hi_f = bootstrap_ci(accs_ft)
    return {
        "num_evaluated": done,
        "acc_gen": round(sum(accs_gen) / done, 4),
        "acc_gen_ci95": [round(lo_g, 4), round(hi_g, 4)],
        "acc_ftok": round(sum(accs_ft) / done, 4),
        "acc_ftok_ci95": [round(lo_f, 4), round(hi_f, 4)],
        "none_gen_rate": round(sum(r["pred_gen"] is None for r in records) / done, 4),
        "none_ftok_rate": round(sum(r["pred_ftok"] is None for r in records) / done, 4),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="MMLU simulated-quant accuracy eval (CPU)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--qparams", required=True, help="calibrate.py 的 .qparams.pt")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--configs", default="minmax,nac",
                    help="逗号分隔，按序执行：minmax,nac,fp16")
    ap.add_argument("--num-samples", type=int, default=1000,
                    help="与板上相同：1000 → 57 科目 × 17 = 969 题")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip", type=int, default=0,
                    help="每配置跳过前 N 题（断点续跑）")
    ap.add_argument("--limit", type=int, default=None, help="调试：只评前 N 题")
    ap.add_argument("--fold", default=None,
                    help="AWQ 折叠参数 .fold.pt（awq_calibrate.py 产物）；qparams 配置评测前折叠，评测后撤销 norms")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    print(f"[sim-mmlu] loading {args.model} (fp32) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    rows = load_mmlu(args.num_samples)  # seed=42，与板上完全同题集
    if args.limit:
        rows = rows[: args.limit]
    print(f"[sim-mmlu] {len(rows)} questions (board-protocol aligned, zs) "
          f"gs={args.group_size} bs={args.batch_size} configs={args.configs}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"model": args.model, "group_size": args.group_size,
                     "num_samples": args.num_samples, "max_new_tokens": args.max_new_tokens,
                     "protocol": "zero-shot generative(32tok)+firsttok, batched sim fp32",
                     "configs": {}}
    fold = None
    if args.fold:
        fold = torch.load(args.fold, map_location="cpu", weights_only=True)
        print(f"[sim-mmlu] fold params loaded from {args.fold} ({len(fold)} blocks)", flush=True)

    for spec in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if "=" in spec:
            cfg, qp_file = spec.split("=", 1)  # 语法: <label>=<qparams.pt>（gptq 等通用）
        else:
            cfg, qp_file = spec, args.qparams
        print(f"[sim-mmlu] ===== config: {cfg} =====", flush=True)
        rec_path = out_path.with_suffix(f".records.{cfg}.json")
        records: list = []
        start = args.skip
        if rec_path.exists() and start == 0:
            # 断点续跑：沿用已有逐题记录
            records = json.loads(rec_path.read_text())
            start = len(records)
            print(f"[sim-mmlu] resume {cfg} from #{start}", flush=True)
        if cfg == "fp16":
            pass
        elif cfg == "minmax":
            apply_qparams(model, args.group_size, "minmax", backup_path=str(BACKUP))
        else:  # nac 及任意 label=qparams 配置：同一 qparams 数值路径
            if fold is not None:
                apply_fold(model, fold)  # AWQ：先折叠生产者/消费者权重再按 qparams 伪量化
            qparams = torch.load(qp_file, map_location="cpu", weights_only=True)
            apply_qparams(model, args.group_size, "nac", qparams, backup_path=str(BACKUP))
        try:
            stats = eval_config(model, tok, rows, args.batch_size,
                                args.max_new_tokens, records, start, out_path, rec_path)
        finally:
            if cfg != "fp16" and BACKUP.exists():
                restore_from_disk(model, str(BACKUP))
            if fold is not None and cfg not in ("fp16", "minmax"):
                undo_fold_norms(model, fold)  # linears 已由 restore_from_disk 精确还原
        results["configs"][cfg] = stats
        print(f"[sim-mmlu] {cfg}: {json.dumps(stats)}", flush=True)
        out_path.with_suffix(".full.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False))

    # 配置间配对差（核心指标）：nac vs minmax，同题配对 + McNemar 不一致对
    def load_acc(cfg):
        p = out_path.with_suffix(f".records.{cfg}.json")
        if not p.exists():
            return None
        recs = json.loads(p.read_text())
        return recs, [r["pred_gen"] == r["gold"] for r in recs]

    if "minmax" in args.configs and "nac" in args.configs:
        mm = load_acc("minmax")
        nc = load_acc("nac")
        if mm and nc:
            n = min(len(mm[0]), len(nc[0]))
            ok_mm, ok_nc = mm[1][:n], nc[1][:n]
            b = sum(1 for a, c in zip(ok_mm, ok_nc) if a and not c)  # minmax 对 nac 错
            c_ = sum(1 for a, c in zip(ok_mm, ok_nc) if not a and c)  # nac 对 minmax 错
            results["paired_nac_vs_minmax"] = {
                "n": n,
                "acc_diff_pp": round((sum(ok_nc) - sum(ok_mm)) / n * 100, 2),
                "mcnemar_b_minmax_only": b, "mcnemar_c_nac_only": c_,
            }
            out_path.with_suffix(".full.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False))
            print(f"[sim-mmlu] paired nac-minmax: {results['paired_nac_vs_minmax']}", flush=True)
    print(f"[sim-mmlu] -> {out_path.with_suffix('.full.json')}", flush=True)


if __name__ == "__main__":
    main()
