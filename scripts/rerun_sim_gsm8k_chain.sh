#!/usr/bin/env bash
# A 线重跑链：sim GSM8K 五配置 dual-metric 重跑（2026-10-29 m1 复核处置，用户拍板）。
#
# 背景：records.gen 历史上截断 300 字符 + 单一抽取规则（取末数字），被
# results/extractor_recheck.json 证明是全配置一阶混杂。本链用升级后的
# eval_gsm8k_sim.py（全文落盘 + acc/acc_answeris 双指标）重跑，产出论文级数字。
# 旧 results 不覆盖，全部写 *_rerun* 新文件。
#
# 前置条件：R4-e 板端流水线结束后才可启动（本链 taskset 4-7 占 A72，
# 与 NPU 评测可共存但会互相拖慢；统一等 R4-e 判别分出来再跑）。
#
# 用法（板上）：
#   cd /media/pi/DATA/rknn-llm/nac_paper
#   nohup bash scripts/rerun_sim_gsm8k_chain.sh > results/rerun_sim_gsm8k.log 2>&1 &
#
# 每跑 ~10-11h × 5 轮 ≈ 55h；断点续跑：eval 脚本自带 records resume，直接重启本链即可。
set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MODEL=models/Qwen2.5-1.5B-Instruct-hf
LOG=results/rerun_sim_gsm8k_chain.status

if [ -f results/.rerun_sim_gsm8k.lock ]; then
  echo "[chain] lock exists (PID $(cat results/.rerun_sim_gsm8k.lock)), abort"; exit 1
fi
echo $$ > results/.rerun_sim_gsm8k.lock
trap 'rm -f results/.rerun_sim_gsm8k.lock' EXIT

: > "$LOG"
run() { # name, out, extra args...
  local name="$1" out="$2"; shift 2
  echo "[chain] $(date '+%F %T') START $name -> $out" | tee -a "$LOG"
  taskset -c 4-7 "$PY" scripts/eval_gsm8k_sim.py \
    --model "$MODEL" --group-size 64 --batch-size 16 \
    --out "$out" "$@" >> "results/rerun_${name}.log" 2>&1
  local rc=$?
  echo "[chain] $(date '+%F %T') DONE  $name rc=$rc" | tee -a "$LOG"
}

# 顺序：fp16 参照点（Table 2 缺失项）→ minmax+nac → gptq → nacv2 → awq
run fp16    results/acc_gsm8k_sim_g64_rerun_fp16.json  --configs fp16            --qparams results/nac_qwen1.5b_g64.qparams.pt
run mmnac   results/acc_gsm8k_sim_g64_rerun.json        --configs minmax,nac      --qparams results/nac_qwen1.5b_g64.qparams.pt
run gptq    results/acc_gsm8k_sim_g64_rerun_gptq.json   --configs gptq            --qparams results/gptq_qwen1.5b_g64.qparams.pt
run nacv2   results/acc_gsm8k_sim_g64_rerun_nacv2.json  --configs nacv2           --qparams results/nacv2_qwen1.5b_g64.qparams.pt
run awq     results/acc_gsm8k_sim_g64_rerun_awq.json    --configs awq             --qparams results/awq_qwen1.5b_g64.qparams.pt --fold results/awq_qwen1.5b_g64.fold.pt

echo "[chain] $(date '+%F %T') ALL DONE" | tee -a "$LOG"
