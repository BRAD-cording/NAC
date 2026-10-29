#!/bin/bash
# sim 轴重跑（Q1 修订所需）：等板端 512 队列结束后启动，避免抢核。
# 协议：2-shot CoT，512 token，停止串 "\nQuestion:"，双指标，全文归档。
# 顺序：fp16（缺失锚点）→ gptq（关键对照）→ nacv2（修复版）
set -u
cd /media/pi/DATA/rknn-llm/nac_paper || exit 1
PY=.venv/bin/python
MARK=results/q1_board512.DONE
LOG=results/q1_sim_queue.log

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 等板端队列 DONE（避免与 NPU 评测抢 A72）
while [ ! -f "$MARK" ]; do sleep 120; done
log "board queue DONE, sim queue start"

run() {
  cfg="$1"; qp="$2"
  out="results/acc_gsm8k_sim_g64_stop_${cfg}.json"
  if [ -f "$out" ]; then log "skip $cfg (exists)"; return; fi
  log "start $cfg"
  taskset -c 4-7 timeout 172800 $PY scripts/eval_gsm8k_sim.py \
      --model models/Qwen2.5-1.5B-Instruct-hf \
      --qparams "$qp" --group-size 64 --configs "$cfg" \
      --num-samples 200 --max-new-tokens 512 --batch-size 16 \
      --stop-strings '\nQuestion:' \
      --out "$out" > "results/q1_sim_${cfg}.log" 2>&1
  log "end $cfg rc=$?"
}

run fp16  results/nac_qwen1.5b_g64.qparams.pt
run gptq  results/gptq_qwen1.5b_g64.qparams.pt
run nacv2 results/nacv2_qwen1.5b_g64.qparams.pt

log "sim queue ALL DONE"
touch results/q1_sim.DONE
