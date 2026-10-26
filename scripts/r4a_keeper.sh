#!/usr/bin/env bash
# r4a keeper v3：双职能
#  (a) 每 30s WSL 客户端心跳，防 WSL VM idle 回收（01:25 事故根因）；
#  (b) 每 60s 活性检查（PID 文件判据，防 pgrep 自匹配假象）：
#      进程消失且无 DONE → schtasks 重启（Windows 持有前台进程，规避 wsl 瞬态后台竞态）。
set -u
LOG=/media/pi/DATA/rknn-llm/nac_paper/results/r4a_keeper.log
SSH="sshpass -p 131468 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 Administrator@192.168.3.16"
ts() { date '+%F %T'; }
tick=0
echo "[$(ts)] r4a-keeper v3 start" >> "$LOG"
while :; do
    $SSH "wsl -d Ubuntu -- echo ok" >/dev/null 2>&1
    tick=$((tick+1))
    if [ $((tick % 2)) -eq 0 ]; then
        # 活性：远端 check_r4a.sh 判据（规避多层转义误判，10-27 事故教训）
        state=$($SSH "wsl -d Ubuntu -- bash /home/dministrator/r3x86/check_r4a.sh" 2>/dev/null | tr -d '\r\000' | grep -vE '^(id:|ssh:|wsl:|your )')
        alive=$(echo "$state" | head -1)
        tail1=$(echo "$state" | tail -1)
        case "$tail1" in *"[r4a] DONE"*) : ;; *)
            if [ "$alive" != "ALIVE" ]; then
                echo "[$(ts)] build dead (tail: ${tail1:0:60}) -> schtasks restart" >> "$LOG"
                $SSH "schtasks /run /tn R4A_BUILD" >/dev/null 2>&1
                sleep 25
                a2=$($SSH "wsl -d Ubuntu -- bash /home/dministrator/r3x86/check_r4a.sh" 2>/dev/null | tr -d '\r\000' | head -1)
                echo "[$(ts)] restart check $a2" >> "$LOG"
            fi ;;
        esac
    fi
    sleep 30
done
