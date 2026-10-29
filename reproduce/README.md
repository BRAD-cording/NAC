# Reproducing the measurements

This directory documents how every number in `paper/draft_v0.1.md` was produced and what is still
missing. It is deliberately explicit about gaps: a reproduction package that hides them is worse than
none.

## 1. Environment

**Board** — NanoPi M5 (Rockchip RK3576): 4×Cortex-A53 @2.016 GHz, 4×Cortex-A72 @2.208 GHz, 16 GB
LPDDR5, 2-core NPU @950 MHz. Runtime `librkllmrt.so` 1.2.3 (from rknn-llm `release-v1.2.3`), driver
rknpu 0.9.8 via `/dev/dri/renderD129`.

**Python** — `.venv` with Python 3.11, torch 2.14.1+cpu (aarch64), transformers 5.18, datasets 5.1.
Exact versions in `results/env_snapshot.json`.

**Fixed frequencies** — `sudo .venv/bin/python -m src.utils.freq --mode performance`
(CPU 2016/2208 MHz, NPU 950 MHz, DDR 2736 MHz). Verify before any timed run; the perf table was
measured at these settings.

## 2. Artifacts

`MANIFEST.md` lists every `.rkllm` with size, md5 and the `model_dtype` reported by the runtime.

Read it before comparing anything. Three artifacts named `nac_qwen1.5b_g64_nac`,
`nac_qwen1.5b_g64_mmctrl` and `nacv2_qwen1.5b_g64_intzp` have **identical md5** (`3bd8c0ba…`): they
are one file, produced by a build in which `extra_qparams` was silently ignored. They differ in dtype
from the vendor reference (`W4A16_G64` vs `W4A16`).

Missing: `Qwen2.5-3B-Instruct_RK3576_w4a16.rkllm` (2.39 GB) was deleted to free space after its
evaluation. Its results (MMLU 0.6037, GSM8K 0.6700 at 512 tokens) cannot be re-run without
re-downloading from the model hub; the sha256 is recorded in `docs/BOARD_NOTES.md`.

## 3. Board accuracy

```bash
# MMLU, 969 items, zero-shot, 32-token cap (single letter — cap-insensitive)
env -u LD_LIBRARY_PATH .venv/bin/python -m src.eval.accuracy \
    --model models/<artifact>.rkllm --task mmlu --num-samples 1000 \
    --fewshot-num 0 --max-new-tokens 32 --out results/acc_mmlu_<tag>_zs.json

# GSM8K, 200 items, zero-shot, 512-token cap (MUST be 512; 32 truncates chain-of-thought)
env -u LD_LIBRARY_PATH .venv/bin/python -m src.eval.accuracy \
    --model models/<artifact>.rkllm --task gsm8k --num-samples 200 \
    --fewshot-num 0 --max-new-tokens 512 --out results/acc_gsm8k_<tag>_zs512.json
```

`--max-new-tokens 32` on GSM8K is not a valid measurement: it truncates every chain of thought
before the answer and drives even the vendor model to 0.0000 (see draft §6.2.1).

Records are archived with full text (4000 chars), `n_tokens`, `hit_cap`, and both extraction rules.

## 4. Simulation (x86 fp32)

```bash
# calibration (example: NAC-v2, g64)
.venv/bin/python scripts/nac_v2_calibrate.py --model models/Qwen2.5-1.5B-Instruct-hf \
    --group-size 64 --out results/nacv2_qwen1.5b_g64

# GSM8K under pseudo-quantization, 2-shot CoT, 512 tokens
taskset -c 4-7 .venv/bin/python scripts/eval_gsm8k_sim.py \
    --model models/Qwen2.5-1.5B-Instruct-hf --qparams results/nac_qwen1.5b_g64.qparams.pt \
    --group-size 64 --configs fp16,minmax,nac --out results/acc_gsm8k_sim_g64.json

# perplexity
taskset -c 4-7 .venv/bin/python scripts/ppl_coherence.py --num-seqs 64 --seqlen 1024 \
    --group-size 64 --modes fp16,minmax,nac,gptq --out results/ppl_g64_64seq.json
```

Rough cost on the board: 10–11 h per 200-item GSM8K configuration, 40 min per MMLU configuration,
1.5–2 h per 64-sequence perplexity.

## 5. Statistics and auditing

```bash
.venv/bin/python scripts/compute_stats_pairs.py      # McNemar + bootstrap, results/stats_pairs.json
.venv/bin/python analysis/check_claims.py            # re-derives every number in the draft
.venv/bin/python analysis/mmlu_audit.py              # extraction-rule sensitivity
.venv/bin/python analysis/gsm8k_audit.py             # length / truncation signatures
.venv/bin/python scripts/q1_report.py                # board 512-token summary
```

`analysis/check_claims.py` exits non-zero if any archived number drifts from the value quoted in the
paper. It is the guard against the failure mode that produced the retracted claims: documents
diverging from the JSON they cite.

## 6. Known gaps

| Gap | Effect | Fix |
|---|---|---|
| 3B artifact deleted | 3B control not re-runnable locally | re-download (2.4 GB) and verify sha256 |
| External injection never demonstrated at 1.5B | no valid board-side NAC arm | rebuild with the flat state-dict schema and verify md5 changes |
| Sim GSM8K archives truncate at 300 chars | extraction sensitivity can only be bounded | re-run with full archive + stop sequence |
| No fp16 simulation GSM8K anchor | quantization loss vs stack loss not separable | one 10 h run |
| Single platform / single model family | external validity unproven | second model family |
