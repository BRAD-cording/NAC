#!/usr/bin/env bash
# 一键重建链（2026-10-09）：qemu x86 环境 → g64 build → 原自动化队列（g128 导出 + NPU 评测）
# 背景：/tmp/rktoolkit 被 tmp 清理机制抹掉，g64 build 于 Optimizing 阶段被杀；环境改放 /media/pi/DATA/rktoolkit
# 用法: nohup bash scripts/rebuild_and_run.sh > results/rebuild.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export RKT_BASE="${RKT_BASE:-/media/pi/DATA/rktoolkit}"

command -v debootstrap >/dev/null 2>&1 || sudo apt-get install -y debootstrap
sudo -n true 2>/dev/null || { echo "需要免密 sudo"; exit 1; }

if [ -x "$RKT_BASE/x86py/bin/python3" ]; then
    echo "[rebuild] $(date '+%F %T') ==> [1/3] qemu 环境已存在，跳过重建"
else
    echo "[rebuild] $(date '+%F %T') ==> [1/3] qemu 环境重建（~30-45min）"
    if ! bash scripts/setup_qemu_x86.sh > results/setup_qemu_x86.log 2>&1; then
        echo "[rebuild] FAIL: setup，看 results/setup_qemu_x86.log"; exit 1
    fi
fi

echo "[rebuild] $(date '+%F %T') ==> [2/3] g64 build（optim 0 跳过病态 Optimizing）"
if ! QEMU_LD_PREFIX="$RKT_BASE/sysroot" "$RKT_BASE/x86py/bin/python3" scripts/build_rkllm_x86.py \
        --model models/nac_dequant_g64 --dataset data/quant_dataset_32.json \
        --dtype w4a16_g64 --optim 0 --out models/nac_qwen1.5b_g64_nac.rkllm \
        > results/build_g64_nac.log 2>&1; then
    echo "[rebuild] FAIL: g64 build，看 results/build_g64_nac.log"; exit 1
fi
grep -q "\[build\] OK" results/build_g64_nac.log || { echo "[rebuild] FAIL: 未见 build OK 标记"; exit 1; }

echo "[rebuild] $(date '+%F %T') ==> [3/3] 接评测队列"
RKT_BASE="$RKT_BASE" bash scripts/export_queue.sh > results/queue.log 2>&1
echo "[rebuild] $(date '+%F %T') ==> 全链完成"
