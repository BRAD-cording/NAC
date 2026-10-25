#!/usr/bin/env bash
# E3/E4 编排 watcher：等 GSM8K sim×3 完成后，停队列腾出内存窗口，
# 依次跑 E4 ppl_coherence → E3 构建+板评，然后恢复队列（断点续跑）。
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> results/mmctrl_watcher.log 2>&1
echo "=== watcher start $(date '+%F %T') ==="

# 1) 等待 GSM8K sim×3 完成（队列 [2/3] DONE 标记）
while ! grep -qF '[2/3] GSM8K g64 3-configs DONE' results/p0_queue.log 2>/dev/null; do
    sleep 300
done
echo "=== gsm8k sim x3 done -> stop queue $(date '+%T') ==="
systemctl --user stop nac-queue 2>/dev/null
sleep 10

# 2) E4: ppl_coherence（干净内存窗口，~35min）；脚本已常驻 scripts/，不再依赖 /tmp
[ -f scripts/ppl_coherence.py ] || { echo "!!! scripts/ppl_coherence.py missing"; exit 1; }
.venv/bin/python scripts/ppl_coherence.py >> results/ppl_coherence.log 2>&1
echo "=== ppl coherence done $(date '+%T') ==="

# 3) E3: minmax 对照构建 + 板端 MMLU zs + GSM8K zs
bash scripts/mmctrl_pipeline.sh

# 4) 恢复队列（GPTQ g128 校准从头跑，损失 <5min）
systemctl --user reset-failed nac-queue 2>/dev/null || true
systemd-run --user --unit=nac-queue --working-directory="$PWD" bash scripts/run_p0_queue.sh
echo "=== queue resumed $(date '+%F %T') ==="
