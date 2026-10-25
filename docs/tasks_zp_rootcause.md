# Tasks: NAC 板端 GSM8K 崩溃根因甄别

对应 spec: docs/spec_zp_rootcause.md

> **进度注记（2026-10-18）**：T2 部分结果已出——sim GSM8K g64：nac **0.070**（200/200 落盘）与 minmax **0.075** 同崩，gptq 0.5625 显著正常 → 判定升级为**双问题叠加**：问题 A = NAC 校准目标缺误差补偿（sim 层面即现，与 NPU 无关）；问题 B = NPU float-zp 重解释嫌疑（由 E3 板端对照裁决，mmctrl-wd 自动执行中）。问题 A 修复已实施：NAC-v2（src/nac/hessian.py，Hessian 补偿+激活加权+int-zp，nac-v2-wd 排队等内存窗口）；gptq sim 评测进行中。E3/E4 与 v2 链结果落地后回填 T5 判别矩阵。

## T1. E1 zp 分布分析（零内存，立即执行）✅ 已完成 2026-10-16 11:40
- 文件: results/zp_analysis.json
- 结论: **H1 前提坐实**——NAC zp 仅 0.16% 为整数（|frac| 均值 0.25 = 完全连续分布，范围 [-0.063, 15.105]，zp_max 超出 4bit 上限 15）；GPTQ zp 100% 整数 [0,15]。若 NPU 取整/截断 zp，等效权重误差均值 0.00213 = 典型权重幅度（0.02）的 **10.7%**（最大 0.077 ≈ 3.9× 权重幅度）——足以摧毁 GSM8K 精确算术而 MMLU 靠四选一容错幸存，与观测完全吻合

## T2. E2 GSM8K sim×3 监控（队列自动，零干预）
- 文件: results/acc_gsm8k_sim_g64.records.json（队列产出）
- 内容: fp16 完成后队列自动跑；完成后按数组位置配对计算 minmax/nac/gptq 三配置 acc 与 McNemar
- 验收: NAC 崩→H3 坐实；正常→转 E3

## T3. E3 minmax 对照构建脚本（✅ 脚本已备齐，✅ 已编排进 watcher 自动执行）
- 文件: scripts/gen_minmax_qparams.py（✅ 已跑完）→ /tmp/minmax_g64_full.qparams（196 条目 164MB，zp float 连续 [0.445,14.725] 与 NAC 同语义）；scripts/mmctrl_pipeline.sh（构建+MMLU zs+GSM8K zs）；scripts/mmctrl_watcher.sh（单元 mmctrl-wd 已挂载 11:46，等 GSM8K sim×3 完成后自动：停队列→E4→E3→恢复队列）
- 内容: ①对 fp32 原始模型按 minmax 生成全量 196 层 qparams（torch.save 格式同 nac 注入）→ /tmp/minmax_g64_full.qparams；②build(w4a16_g64, normal, extra_qparams=...) → models/nac_qwen1.5b_g64_mmctrl.rkllm；③板端评测 MMLU zs + GSM8K zs
- 约束: 必须在 GSM8K sim×3 完成后的窗口执行（约 10-17 凌晨），或用户批准牺牲 MMLU g128 段
- 验收: 对照板评分数落盘，判别矩阵填毕

## T4. E4 ppl_coherence 重跑（✅ 已编排进 watcher，E3 构建前自动执行）
- 文件: /tmp/ppl_coherence.py → watcher 自动 cp 至 scripts/ppl_coherence.py → results/ppl_coherence.log
- 验收: 16seq PPL 对照 10-08 值（fp16 10.627/minmax 11.738/nac 11.408），偏差 <0.5% 视为代码无回归

## T5. 根因落定与修复决策
- 内容: 汇总 E1-E4 判别矩阵 → 按 spec §4 选择修复路径；若 integer-zp 重校准则改 src/nac/quantize.py（zp 取整格点）并重跑校准
- 验收: docs/spec_zp_rootcause.md 回填判别矩阵与决策

## 执行顺序
T1（立即）→ T2（等待队列，被动）→ T3/T4（夜间窗口）→ T5
