#!/bin/bash
# Q1 决定性复测（v2）：板端 512-token，只跑真正互异的产物
# 依据：nac / mmctrl / nacv2intzp 三个 .rkllm 的 md5 全为 3bd8c0ba…（注入从未生效，同一个 qemu 构建）
# 因此只跑 3 个互异产物 + 1 个锚点：
#   bd8   = nac_qwen1.5b_g64_nac.rkllm      (3bd8c0ba, 注入无效的 plain qemu 构建, MMLU 0.5614)
#   symv2 = sym_minmax_g64_v2.rkllm         (d6ff15, 唯一互异的 minmax 注入候选)
#   vendor= Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm (90330bf1, 厂商基线锚点)
set -u
cd /media/pi/DATA/rknn-llm/nac_paper || exit 1
PY=.venv/bin/python
log() { echo "[$(date '+%F %T')] $*" >> results/q1_board512_queue.log; }

while pgrep -f "src.eval.accuracy" > /dev/null 2>&1; do sleep 60; done
log "queue v2 start"

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
  if [ $rc -ne 0 ]; then cp "results/q1_512_${tag}.log" "results/q1_512_${tag}.FAIL.log" 2>/dev/null; fi
}

run q1bd8    models/nac_qwen1.5b_g64_nac.rkllm
run q1symv2  models/sym_minmax_g64_v2.rkllm
run q1vendor models/Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm

log "queue v2 ALL DONE"
touch results/q1_board512.DONE
