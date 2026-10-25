#!/usr/bin/env bash
# R3 x86 原生构建启动脚本（在用户的 i7 笔记本上运行，勿在板子上跑）。
#
# 前置条件：
#   1. Linux x86_64 或 Windows + WSL2（Ubuntu 22.04，自带 python3.10）
#   2. 本目录结构（与板端项目一致）：
#        rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl
#        scripts/r3_plain_build.py
#        data/quant_dataset.json
#        models/Qwen2.5-1.5B-Instruct-hf/   （2.9G，从板端 scp）
#   3. 内存 16G：load fp32 ≈6G + Optimizing 峰值，若 OOM 建议临时加 swap
#
# 产物（构建完成后 scp 回板子）：
#        models/local_w4a16_plain.rkllm   （板端评测用）
#        results/tk_w4a16_full.qparams    （toolkit 自算 qparams，可对比 R1 Schema）
set -euo pipefail
cd "$(dirname "$0")"

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYVER" != "3.10" ]; then
    echo "!!! 需要 python3.10（wheel 是 cp310），当前 python3=$PYVER"
    echo "    Ubuntu 22.04 自带 3.10；其他系统请先装 python3.10 再重跑"
    exit 1
fi

[ -d models/Qwen2.5-1.5B-Instruct-hf ] || { echo "!!! 缺 models/Qwen2.5-1.5B-Instruct-hf"; exit 1; }
[ -f rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl ] || { echo "!!! 缺 wheel"; exit 1; }
[ -f data/quant_dataset.json ] || { echo "!!! 缺 data/quant_dataset.json"; exit 1; }

echo "=== [1/3] venv + 安装 toolkit ==="
python3 -m venv .venv-x86 2>/dev/null || true
.venv-x86/bin/pip install -q rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl

echo "=== [2/3] 原生构建（无 qemu，预计远快于板端） $(date '+%F %T') ==="
mkdir -p results models
.venv-x86/bin/python scripts/r3_plain_build.py

echo "=== [3/3] 完成 $(date '+%F %T')，回传命令： ==="
echo "  scp models/local_w4a16_plain.rkllm results/tk_w4a16_full.qparams \\"
echo "      pi@<板子IP>:/media/pi/DATA/rknn-llm/nac_paper/models/../results/  # 注意分开放：rkllm→models/，qparams→results/"
