#!/usr/bin/env bash
# 自动化链（2026-10-09）：
#   ① 等待 g64 qemu build（已在后台单独跑）→ NPU 冒烟
#   ② 清磁盘 → g128 gen-dequant（ARM 原生）→ qemu build → NPU 冒烟
#   ③ NPU 评测队列：两模型 zero-shot 精度（MMLU 969 + GSM8K 200）→ perf → 阈值扫描
# 每步独立日志在 results/queue_*.log；任一步失败即中止并打 FAIL。
# 用法: nohup bash scripts/export_queue.sh > results/queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

RKT="${RKT_BASE:-/media/pi/DATA/rktoolkit}"
PY="$PWD/.venv/bin/python"
X86PY="$RKT/x86py/bin/python3"
export HF_HUB_OFFLINE=1
RUN() { taskset -c 4-7 env OMP_NUM_THREADS=4 "$@"; }

G64=models/nac_qwen1.5b_g64_nac.rkllm
G128=models/nac_qwen1.5b_g128_nac.rkllm
QLOG=results/queue.log

fail() { echo "[queue] $(date '+%F %T') FAIL: $1" | tee -a "$QLOG"; exit 1; }
step() { echo "[queue] $(date '+%F %T') ==> $1" | tee -a "$QLOG"; }

sudo -n true 2>/dev/null || fail "需要免密 sudo（定频用）"

# ① 等待 g64 build（最长 60min）
step "等待 g64 build 完成"
ok=""
for i in $(seq 1 120); do
    grep -q "\[build\] OK" results/build_g64_nac.log 2>/dev/null && { ok=1; break; }
    pgrep -f build_rkllm_x86.py >/dev/null 2>&1 || break
    sleep 30
done
[ -n "$ok" ] || fail "g64 build 未成功结束（看 results/build_g64_nac.log）"
[ -s "$G64" ] || fail "$G64 不存在或为空"
step "g64 build OK ($(du -h "$G64" | cut -f1))"

step "g64 NPU 冒烟"
RUN "$PY" scripts/smoke_test.py --model "$G64" > results/queue_smoke_g64.log 2>&1 \
    || fail "g64 冒烟失败（看 results/queue_smoke_g64.log）"
grep -q "PASS" results/queue_smoke_g64.log || fail "g64 冒烟未见 PASS"

step "清理 dequant g64（腾 g128 空间）"
rm -rf models/nac_dequant_g64

# ② g128 导出链
step "g128 gen-dequant（ARM 原生）"
RUN "$PY" -m src.nac.export_rkllm --mode gen-dequant \
    --model models/Qwen2.5-1.5B-Instruct-hf \
    --qparams results/nac_qwen1.5b_g128.qparams.pt \
    --group-size 128 --out models/nac_dequant_g128 \
    > results/queue_dequant_g128.log 2>&1 \
    || fail "g128 gen-dequant 失败（看 results/queue_dequant_g128.log）"
[ -s models/nac_dequant_g128/model.safetensors ] || fail "g128 dequant 产物缺失"

step "g128 qemu build"
QEMU_LD_PREFIX="$RKT/sysroot" "$X86PY" scripts/build_rkllm_x86.py \
    --model models/nac_dequant_g128 --dataset data/quant_dataset.json \
    --dtype w4a16_g128 --out "$G128" > results/build_g128_nac.log 2>&1 \
    || fail "g128 build 失败（看 results/build_g128_nac.log）"

step "g128 NPU 冒烟"
RUN "$PY" scripts/smoke_test.py --model "$G128" > results/queue_smoke_g128.log 2>&1 \
    || fail "g128 冒烟失败（看 results/queue_smoke_g128.log）"
grep -q "PASS" results/queue_smoke_g128.log || fail "g128 冒烟未见 PASS"

step "清理 dequant g128"
rm -rf models/nac_dequant_g128

# ③ NPU 评测队列（先精度后性能，重要数据优先落盘）
step "定频 performance"
sudo "$PY" -m src.utils.freq --mode performance >> "$QLOG" 2>&1 \
    || fail "定频失败"

for GS in g64 g128; do
    M=models/nac_qwen1.5b_${GS}_nac.rkllm
    step "精度 zero-shot: $GS（MMLU 969 + GSM8K 200，~1.7h）"
    RUN "$PY" -m src.eval.accuracy --model "$M" --task both \
        --fewshot-num 0 --num-samples 1000 \
        --out "results/acc_${GS}_nac_zs.json" \
        > "results/queue_acc_${GS}_nac_zs.log" 2>&1 \
        || fail "精度评测失败: $GS（看 results/queue_acc_${GS}_nac_zs.log）"
    step "perf: $GS（128/512/1024/2048 含功耗）"
    RUN "$PY" -m src.eval.perf --model "$M" --tokenizer models/tokenizer --power \
        --out "results/perf_${GS}_nac.json" \
        > "results/queue_perf_${GS}_nac.log" 2>&1 \
        || fail "perf 失败: $GS"
    step "阈值扫描: $GS"
    RUN "$PY" -m src.routing.threshold_scan --model "$M" \
        --tokenizer models/tokenizer \
        --out "results/route_threshold_${GS}_nac.json" \
        > "results/queue_route_${GS}_nac.log" 2>&1 \
        || fail "阈值扫描失败: $GS"
done

step "全部完成"
