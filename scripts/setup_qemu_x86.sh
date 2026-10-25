#!/usr/bin/env bash
# 重建板上 qemu x86 环境（2026-10-09：原 /tmp/rktoolkit 被 tmp 清理机制抹掉，build 中途被杀）
# 产物：$RKT_BASE/{x86py, sysroot}；x86py = python-build-standalone cpython3.10 + torch-cpu + transformers + rkllm-toolkit 1.2.3
# 用法: RKT_BASE=/media/pi/DATA/rktoolkit bash scripts/setup_qemu_x86.sh
set -euo pipefail
RKT_BASE=${RKT_BASE:-/media/pi/DATA/rktoolkit}
mkdir -p "$RKT_BASE"
cd "$RKT_BASE"

# 1) x86 python（NJU github-release 镜像，取 3.10 最新 install_only）
if [ ! -x x86py/bin/python3 ]; then
    echo "==> [1/4] 下载 python-build-standalone cpython3.10 x86_64"
    F=$(curl -fsSL "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/LatestRelease/" \
        | grep -o 'cpython-3\.10\.[0-9+]*-x86_64-unknown-linux-gnu-install_only\.tar\.gz' | head -1)
    [ -n "$F" ] || { echo "未找到 PSA 3.10 包名"; exit 1; }
    echo "    $F"
    curl -fL --retry 3 -o psa.tar.gz "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/LatestRelease/$F"
    mkdir -p x86py && tar -xzf psa.tar.gz -C x86py --strip-components=1 && rm psa.tar.gz
fi

# 2) sysroot（确定性构建：直接解包基础 deb，不再用 debootstrap）
if [ ! -d sysroot/usr/lib/x86_64-linux-gnu ]; then
    echo "==> [2/4] 构建 amd64 sysroot（deb 直解）"
    bash "$(dirname "$0")/setup_sysroot_debs.sh"
fi

# 3) qemu 下冒烟 + pip 依赖
echo "==> [3/4] qemu 冒烟 + pip 依赖（较慢，属正常）"
export QEMU_LD_PREFIX="$RKT_BASE/sysroot"
P="$RKT_BASE/x86py/bin/python3"
"$P" -V
"$P" -m pip install --no-cache-dir -q -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"$P" -m pip install --no-cache-dir -q torch --index-url https://download.pytorch.org/whl/cpu
"$P" -m pip install --no-cache-dir -q "transformers==4.55.2" "numpy==1.26.4" pyyaml tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4) rkllm-toolkit 1.2.3 x86_64 wheel
# （2026-10-09：GitHub 直连不可达，wheel 已从本地 git tag release-v1.2.3 提取；
#   元数据钉死 torch==2.6.0/vision 等会被 PyPI 解析成 CUDA 全家桶 ~3G 撑爆磁盘，
#   故 --no-deps 安装 + tuna 装最小依赖，保留已装的 CPU torch 2.14.1——旧环境实证可 build）
echo "==> [4/4] rkllm-toolkit 1.2.3"
W="$RKT_BASE/rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl"
[ -f "$W" ] || { echo "缺 $W（git -C /home/pi/rknn-llm show release-v1.2.3:rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp310-cp310-linux_x86_64.whl > $W）"; exit 1; }
"$P" -m pip install --no-cache-dir -q --no-deps "$W"
"$P" -m pip install --no-cache-dir -q \
    "setuptools==80.9.0" safetensors==0.5.3 sentencepiece==0.2.0 protobuf einops==0.4.1 tiktoken \
    tabulate==0.9.0 "jinja2==3.1.4" flatbuffers==24.3.25 jsonschema \
    colorlog==6.8.2 jsonlines==4.0.0 easydict==1.13 addict==2.4.0 \
    transformers-stream-generator==0.0.5 scipy datasets==4.1.1 pyarrow==21.0.0 accelerate==1.5.2 pillow==10.4.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
"$P" -c "from rkllm.api import RKLLM; print('rkllm_toolkit import OK')"
echo "[setup] OK"
