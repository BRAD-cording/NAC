#!/usr/bin/env bash
# x86 校准机依赖（7B 校准 + toolkit 导出 + 校准期 PPL）
set -euo pipefail

echo "==> 建议 python3.10 venv"
python3.10 -m venv .venv-x86 2>/dev/null || python3 -m venv .venv-x86
source .venv-x86/bin/activate

pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets pyyaml tqdm
pip install ../rkllm-toolkit/packages/rkllm_toolkit-1.1.4-cp310-cp310-linux_x86_64.whl

python - <<'EOF'
from rkllm.api import RKLLM
print("rkllm_toolkit OK")
EOF
echo "==> x86 环境就绪"
