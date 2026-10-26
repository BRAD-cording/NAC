#!/usr/bin/env bash
# R4-a 构建判活（远端执行，规避多层转义）：输出两行 —— ALIVE/DEAD、日志末行
p=$(cat /home/dministrator/r3x86/build_r4a.pid 2>/dev/null)
if [ -n "$p" ] && [ -r "/proc/$p/cmdline" ] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q python; then
    echo ALIVE
else
    echo DEAD
fi
tail -c 300 /home/dministrator/r3x86/build_r4a.log 2>/dev/null | tr -d '\000' | tail -1
