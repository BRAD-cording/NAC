#!/bin/bash
# Q1 决定性复测：板端 512-token 生成协议，四个配置顺序执行（等当前评测退出后启动）
# 目的：验证"板端 GSM8K 崩塌"是否为 32-token 截断伪影（见 docs/review_q1_20261029_v2.md）
set -u
cd /media/pi/DATA/rknn-llm/nac_paper || exit 1
PY=.venv/bin/python
MARK=results/q1_board512.DONE

log() { echo "[$(date '+%F %T')] $*" >> results/q1_board512_queue.log; }

# 1) 等所有在跑的 accuracy 评测结束（含本次会话前启动的 plain_x86@512）
while pgrep -f "src.eval.accuracy" > /dev/null 2>&1; do sleep 60; done
log "queue start (no accuracy process running)"

run() {
  tag="$1"; model="$2"
  out="results/acc_gsm8k_${tag}_zs512.json"
  if [ -f "$out" ]; then log "skip $tag (exists)"; return; fi
  log "start $tag model=$model"
  env -u LD_LIBRARY_PATH timeout 21600 $PY -m src.eval.accuracy \
      --model "$model" --task gsm8k --num-samples 200 --fewshot-num 0 \
      --max-new-tokens 512 --out "$out" > "results/q1_512_${tag}.log" 2>&1
  rc=$?
  log "end $tag rc=$rc"
  if [ $rc -ne 0 ]; then
    cp "results/q1_512_${tag}.log" "results/q1_512_${tag}.FAIL.log" 2>/dev/null
  fi
}

run nac512      models/nac_qwen1.5b_g64_nac.rkllm
run nacv2_512   models/nacv2_qwen1.5b_g64_intzp.rkllm
run mmctrl512   models/nac_qwen1.5b_g64_mmctrl.rkllm
run vendor512   models/Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm

log "queue ALL DONE"
touch "$MARK"
