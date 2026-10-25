#!/usr/bin/env bash
# RK3576 端依赖安装（D1 环境就绪）
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> 系统包"
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git build-essential

echo "==> 板载 venv + Python 依赖（numpy/transformers/datasets/torch-cpu）"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
# aarch64 用 CPU 版 torch wheel（镜像源无 aarch64 CUDA 包，直接装也行但更大）
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install -r requirements.txt -q

echo "==> librkllmrt.so 可加载性检查"
python3 - <<'EOF'
import ctypes
from pathlib import Path
lib = Path("../rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so")
assert lib.exists(), f"缺少 {lib}（需 rknn-llm 仓库与 nac_paper 同级）"
ctypes.CDLL(str(lib))
print(f"OK: {lib} 加载成功")
EOF

echo "==> 定频（需 root，评测前必做）"
sudo "$VIRTUAL_ENV/bin/python" -m src.utils.freq --mode performance || true

echo "==> 环境就绪。下一步：python -m src.utils.data 生成校准集"
