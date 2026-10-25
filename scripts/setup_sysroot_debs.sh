#!/usr/bin/env bash
# 确定性 amd64 sysroot 构建（2026-10-09，替代不稳定的 debootstrap）：
# 从 tuna 的 bullseye Packages 索引解析基础包 → 下载 .deb → 解包进 sysroot → 修 lib64 loader 软链
# 用法: sudo 权限可用即可；RKT_BASE=/home/pi/rktoolkit bash scripts/setup_sysroot_debs.sh
set -euo pipefail
RKT_BASE="${RKT_BASE:-/home/pi/rktoolkit}"
R="$RKT_BASE/sysroot"
MIRROR="https://mirrors.tuna.tsinghua.edu.cn/debian"
SUITE="bullseye"
PKGS="libc6 libgcc-s1 libstdc++6 zlib1g libgomp1 libcrypt1"

[ -f "$R/etc/debian_version" ] && [ -d "$R/usr/lib/x86_64-linux-gnu" ] && {
    echo "[sysroot] 已就绪，跳过"; exit 0; }

mkdir -p "$RKT_BASE/debs" && cd "$RKT_BASE/debs"

echo "==> 拉取 $SUITE main amd64 包索引"
curl -fsSL --retry 3 -o Packages.gz "$MIRROR/dists/$SUITE/main/binary-amd64/Packages.gz"
gunzip -f Packages.gz

echo "==> 解析 $PKGS"
: > debs.list
for p in $PKGS; do
    f=$(awk -v pkg="$p" '
        /^Package: / { cur=$2 }
        /^Filename: / { if (cur == pkg) { print $2; exit } }
    ' Packages)
    [ -n "$f" ] || { echo "未找到包 $p"; exit 1; }
    echo "$f" >> debs.list
done
cat debs.list

echo "==> 下载并解包"
while read -r f; do
    d="$(basename "$f")"
    [ -s "$d" ] || curl -fsSL --retry 3 -o "$d" "$MIRROR/$f"
    sudo dpkg-deb -x "$d" "$R"
done < debs.list

echo "==> 修 lib64 loader 软链"
[ -f "$R/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2" ] || {
    echo "缺 ld-linux-x86-64.so.2"; exit 1; }
sudo rm -rf "$R/lib64"
sudo mkdir -p "$R/lib64"
sudo ln -sfn ../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 "$R/lib64/ld-linux-x86-64.so.2"
sudo mkdir -p "$R/etc"

echo "==> 生成 ld.so.cache（qemu 下跑 amd64 ldconfig）"
sudo chroot "$R" /sbin/ldconfig 2>/dev/null \
    || echo "WARN: ldconfig 失败（可接受，qemu 自带搜索路径）"

[ -d "$R/usr/lib/x86_64-linux-gnu" ] && echo "[sysroot] OK"
