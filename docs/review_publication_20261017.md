# NAC 论文发表评审级评估报告（2026-10-17）

> 评审对象：`nac_paper` 全项目（方法论、NAC 算法、实验矩阵、论文骨架 `paper/outline.md`）
> 评审基准：Applied Intelligence（Springer，IF 3.5，JCR Q2-AI，一审中位 45 天，来源：springer.com/10489 及 JCR 表）为首选；IEEE Access 为备选
> 前置报告：review_q2_20261015.md（Q2+ 差距分析，10-18 已删除，阻断项处置状态见 TECHNICAL_REPORT 进展日志）；[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)（10-16 底稿）；[spec_zp_rootcause.md](spec_zp_rootcause.md)（根因甄别 spec）
> 动态实验口径：本报告含 10-17 22:22 前 E2 已落盘数据；gptq sim 段进行中（16/200）
> **10-18 补记**：E2 gptq 段收官 **0.305**（200/200 终值；早段 16/200 快照 0.5625 系小样本偏高）——仍 4× 于 minmax 0.075/nac 0.070，双问题判定不变；§六 步骤 5 所列"路由跑/砍、AWQ、PPL 扩样、能效叙事"已全部处置（路由砍、AWQ+PPL 已实施排队、能效收敛延迟/吞吐/内存三轴）；TECHNICAL_REPORT 已升 v1.1 同步（含 NAC-v2 方法论小节）；文档冗余清理完成（10-15 版评审已删除并更新全部引用）。
> **10-18 晚间补记（判定闭环）**：**E4 收官**——PPL 复核 minmax 11.738 / nac 11.408 与 10-08 首测完全一致，排除 qparams 过期/代码漂移混淆；**E3 收官**——minmax float-zp 对照注入板端 GSM8K zs **0.005**（与 NAC 注入 0.005 同崩、失败模式一致）、MMLU 0.5614（平手）→ **问题 B（NPU 重解释 float zp）直接坐实**，"对照也崩"分支落定，双问题叠加从预判变为双证据闭环（问题 A=E2、问题 B=E3）。§六 步骤 1-3 全部完成；剩余仅步骤 4 的结果回收（NAC-v2 与 AWQ 自动链，v2 sim GSM8K 为问题 A 修复验收：须显著超 nac 0.070、追近 gptq 0.305）。过程事故：watcher 恢复队列时 systemd-run 缺 `--working-directory` 致 nac-queue 127 静默失败（停摆 ~14h），已修复并以正确 cwd 重启（10-18 22:42），续跑零损失。
> **10-20 补记（v2 首判 + g128 边界条件）**：MMLU g128 三配置收官（minmax 0.3622 / **nac 0.4293** / gptq 0.4871）——nac vs minmax **+6.71pp 反转**（McNemar p<1e-7），v1 劣化仅存在于 g64（边界条件，TECHNICAL_REPORT §3.1）；**v2 sim GSM8K 终值 0.150**（200/200）——显著超 v1 nac 0.070（b=26/c=10，p=0.012）但显著低于 gptq 0.305（b=15/c=46，p=1.2e-4），**问题 A 修复部分有效、"追近 gptq"未达成**。§四 收录概率表"双问题修复成功"一档判定失效一半：问题 B 修法在位（int-zp），问题 A 仅部分修复。若 AWQ 对照同样不及 gptq，叙事按 §三 预案收敛为"系统评测 + 语义发现 + 部分修复 + 边界刻画"，期刊定位相应下探（IEEE Access 系统评测向权重上调）。余下自动链：v2 MMLU → PPL 64-seq → AWQ 校准+双评测（预计 10-22 下午收官）。
> **10-22 补记（v2 MMLU 收官 + PPL 扩样关账 + AWQ 顺延）**：v2 sim MMLU 终值 **0.4469**（969/969）——显著超 v1 nac 0.3932（b=98/c=46，p=2.1e-5），与 gptq 0.4520 差 −0.51pp **统计平手（p=0.716）**，对 minmax/fp16 亦无显著差（p=0.106/0.13）→ **问题 A 修复在 MMLU 上全额追平 gptq**；结合 GSM8K 0.150 仍显著低于 gptq，v1/v2 与 GPTQ 的残余差距被精确界定在**长程算术单轴**——"短上下文任务全恢复、长程算术部分恢复"，"部分修复 + 边界刻画"证据链完整。PPL 64-seq 扩样落定（P2-7 关账，流血清单 #5 处置）：fp16 11.075 / minmax 12.225 / nac 11.893 / gptq 12.168——v1 NAC PPL 占优与 gptq PPL 最差并存，"任务精度-困惑度"双轴权衡入 §3.1（v2 PPL 已排链尾 stage 9 补齐）。执行事故：AWQ 校准 10-21 20:10 起连环暴露 3 处 bug（hook 键名 KeyError 崩溃；fold σ 形状不匹配崩溃；**生产者 bias 未同步÷scale 的静默数学错误**——Qwen2.5-1.5B qkv 带 bias，2 块冒烟通过仍会毒化全部 AWQ 数据，靠 block 级折叠等价性断言（修复后 rel 6.7e-6 通过）揪出），10-22 凌晨全部修复并验证后幂等重启，追加 v2 PPL 阶段；AWQ 链收官顺延至 **10-23 凌晨**，AWQ sim GSM8K（叙事定判点）预计 10-22 中午出值。
> **10-22 晚间补记（AWQ 收官 + 全矩阵关账 + 叙事定判）**：AWQ sim GSM8K acc=**0.000**（0/64 主动截断）——records 抽查显示生成连贯、CoT 正确但不终止、跨题续写致抽取错位；PPL 分解探针落定：fold_only **10.627**（精确复现 fp16 → 折叠无损）、awq_full **11.226**（16-seq 三方法最优，优于 nac 11.408/minmax 11.738）→ **无实现 bug，EOS 终止退化为真实校准行为**。AWQ sim MMLU **0.4396**（969/969）：与 gptq −1.24pp（p=0.369）、与 v2 −0.72pp（p=0.579）**双平手**，显著低于 minmax（−2.79pp，p=0.033）/fp16（−2.37pp，p=0.038）→ **崩溃为 GSM8K 长生成（EOS）任务特异，首 token/分类任务无损**。与 v1 NAC（PPL 占优 + 任务崩）并案：**激活加权目标族（NAC-v1 截断 / AWQ scale）在 PPL/首 token 轴无损甚至占优、在长程生成轴崩溃**——PPL-下游脱钩升级为方法族现象，问题 A"缺长程补偿"归因获第二族独立证据；v2/GPTQ 的 Hessian 补偿被反衬为长程生成存活的关键分量。v2 PPL 11.962 落定（+0.6% vs v1，近零代价）。**叙事定判**：AWQ/v1/v2 均未及 gptq 长程水平（GSM8K 0.000/0.070/0.150 vs 0.305；MMLU 0.4396/0.3932/0.4469 vs 0.4520），§四 概率表按"部分修复 + 边界刻画"档执行，期刊定位照 10-20 补记预案（IEEE Access 系统评测向权重上调）。**全实验矩阵关账（10-22 21:16）**，余下仅 outline 重写与写作（§六 步骤 6）。

---

## 一、结论（先行）

**方法与统计已达到 Applied Intelligence 平均线，但核心主张的证据链目前在板端与 sim 双侧出现断裂，今天原样投稿大概率被拒（预估 <10% 最终收录）。** E3 根因落定 + 双问题修复（见 §3）完成后，进入 solid Q2 submission 区间（预估 60–70% 经 1–2 轮修回后收录）。没有人能保证收录。

---

## 二、分层评审

### 2.1 已达标资产

| 维度 | 证据 | 评级 |
|---|---|---|
| 算法实现 | NAC grid：GPTQ 式顺序传播 + AWQ 式激活加权 per-group 网格搜索（κ∈{1.0..0.4}），qparams 与 NPU 反量化公式 `W=scale·(q−zp)` 严格一致（`src/nac/calibrate.py`） | 达标 |
| 基线忠实度 | GPTQ 同校准数据/同格式/非对称 uint4 + maxshrink 网格（percdamp 0.01，6527s 实测） | 达标 |
| 统计严谨性 | MMLU 969 题分层、CI95、McNemar 配对（χ² 连续性修正）、断点续跑、增量落盘 | **超过 Q2 平均线** |
| PPL 门槛 | g64：fp16 10.627 / minmax 11.738 / **nac 11.408**；g128：minmax 11.988 / **nac 11.679** | 达标 |
| 板端系统数据 | 1.5B/3B 精度 zs；TTFT 91.5→181.5ms、decode 13.0→7.4 tok/s 四档 prompt 定频 | 部分达标（缺能效） |
| 工程深度 | extra_qparams 注入链路逆向破解、few-shot 崩坏定性、量化器三轴区分方法论 | 可升格为系统发现 |

### 2.2 流血清单（每条足以单独引发 major revision）

1. **板端因果链断裂（最致命）**：板端 NAC GSM8K zs = **0.005 vs 基线 0.555**（1/200 vs 111/200，McNemar p=7e-25）；同模型板端 MMLU zs 0.5614 ≈ 基线 0.5583（p=0.878 平手）。
2. **sim MMLU 劣化**：NAC 0.3932 vs minmax 0.4675（**−7.43pp**，p<1e-4，McNemar b=108/c=36）；fp16 0.4634、gptq 0.4520。"PPL 更优 + 任务精度更差"的组合会被审稿人直问校准目标。
3. **E2 新证据（10-17）：sim GSM8K 三配置三方分化**——gptq 16/200 acc **0.5625**（9/16），minmax 0.075、nac 0.070（同检查点 1/16、2/16）。sim 协议被证明有效（gptq 达板端基线水平），因此 **minmax 与 NAC 的 sim GSM8K 崩溃是真实的校准质量问题**，不是协议假象。原判别矩阵里"NAC 崩 + minmax 正常→H3；全正常→板端特有"之外出现了第三种组合：**minmax 与 NAC 同崩、gptq 独活**。
4. **贡献 2（异构路由）零数据**：代码在、结果零（P0-3 未决，跑或砍）。
5. **P1 级**：AWQ 缺席、能效 tokens/J 无电流采样不可测、PPL 仅 16 seq 样本、7B 缺席。

---

## 三、根因结构更新：从"单问题"到"双问题叠加"

10-16 spec 的工作假设是单一根因（H1：NPU 对 float zp 取整/重解释）。E2 的三方分化要求修正为**双问题叠加假说**：

- **问题 A（校准目标缺陷，sim 可见）**：在 float-zp 忠实语义下，minmax 与 NAC 的 qparams 在 GSM8K 2-shot CoT 上崩溃（~0.07）而 gptq 独活（~0.56）。MMLU 32-token 短生成无法暴露的**长程误差累积**在算术链上被放大。这说明激活加权 MSE + 截断网格的目标函数对多步推理不足，GPTQ 的 Hessian 误差补偿在此场景占优。
- **问题 B（板端 zp 语义，待 E3 裁决）**：NAC zp 99.84% 非整数（|frac| 均值 0.25）vs GPTQ 100% 整数；若 NPU 取整 zp，等效权重误差 ~10.7%（均值）。板端 NAC GSM8K 0.005 的雪上加霜部分。

**对修复路径的影响**：spec §4 原方案"H1 坐实 → integer-zp 约束重校准 → 板端 GSM8K ≥0.50"现在不够——integer-zp 只修问题 B；问题 A 使 NAC 在 sim（忠实语义）下也只有 0.070，**必须同时升级校准目标**（引入 Hessian/GPTQ 型补偿项，或 NAC×GPTQ 混合），并以 sim GSM8K 为快速迭代回路先冲 gptq 水平（~0.55），再做 integer-zp 约束上板。修复工作量比原估计显著增加。

**对论文叙事的影响**：若修复成功，故事反而更强——"长程算术误差暴露了激活加权 MSE 的盲区 + 板端 zp 重解释语义发现 + 双重修复后的 SOTA 恢复"。若 NAC 目标函数升级后仍追不平 gptq，则方法论贡献降级为"系统评测 + 语义发现"，目标期刊建议下调或转 IEEE Access 系统评测向。

---

## 四、评审等级与收录概率

### 4.1 如果今天投出（模拟审稿）

Applied Intelligence：**Reject 或无修回价值的 major revision**。一致性硬伤：方法声称 NPU-aware，板端 GSM8K 归零 + sim MMLU −7.43pp 同时在案，任何话术不可修复。

### 4.2 条件概率（本报告判断，非承诺）

| 状态 | APIN 最终收录概率 | 说明 |
|---|---|---|
| 今天原样投出 | <10% | 一致性硬伤 |
| E3 落定 + 双问题修复成功（板端 GSM8K ≥0.50 且 MMLU ≥0.55）+ P0-2 表格纪律 + P0-3 处置 | **~60–70%** | 经 1–2 轮 major→minor |
| 再加 AWQ 实现或书面取舍 + PPL 扩样 ≥64 seq + 路由数据落地 | ~75% | |
| IEEE Access 备选 | 各状态 +10pp | 系统评测向容错更高 |
| TNNLS/TCAD 级 Q1 | 不现实 | 单平台单模型证据域不足 |

结构性风险不可消除项：单平台（仅 RK3576）、NAC 证据域单模型（1.5B）、3B 仅板端预转基线——只能靠 Limitations 诚实声明 + 3B 定位为"系统评测扩展"化解。

---

## 五、题目评审

现题目：*"NPU-Aware Post-Training Quantization for Efficient LLM Inference on Rockchip RK3576 Edge SoC"*

- **结构合理**：方法 + 具体平台命名，系统类论文惯例，检索友好，长度适中。
- **"NPU-Aware" 目前名不副实**：现有证据表明方法恰恰未感知 NPU 对 zp 的重解释（问题 B），且未感知长程算术误差累积（问题 A）。修复落地后该词才名副其实且更有分量。**投稿前题目里每个形容词必须有数据背书。**
- **"Efficient" 需对齐支撑轴**：能效不可测（无电流采样），全文效率叙事须收敛到延迟/吞吐/内存三轴，或接入 INA260 后再谈 tokens/J。

备选（按最终叙事二选一）：

1. 双重修复叙事成立（推荐）：*"NPU-Aware Post-Training Quantization with Hardware-Faithful Zero-Point Semantics for W4A16 LLM Inference on Rockchip RK3576"*
2. 保守系统评测向：*"Post-Training Quantization for W4A16 LLM Inference on Rockchip RK3576: Calibration Semantics, Board-Level Evaluation, and a Zero-Point Reinterpretation Study"*

---

## 六、关键路径（72 小时 → 投稿）

| 步骤 | 触发/时间 | 内容 |
|---|---|---|
| 1 | 10-18 ~08:00 | gptq sim 200 题收官，E2 判别矩阵闭合（当前 16/200 为强信号，待确认趋势） |
| 2 | 10-18 上午自动 | E4 PPL 复核 → E3 minmax float-zp 对照构建 + 板评 → 问题 B 唯一落定 |
| 3 | E3 后 | 问题 A 修复启动：校准目标升级（Hessian 补偿/混合），sim GSM8K 快速迭代至 ≥0.50 |
| 4 | 并行 | integer-zp 约束版 qparams 重校准 → 注入重建 → 板端复评（GSM8K ≥0.50 且 MMLU ≥0.55，spec §6 标准） |
| 5 | 并行定夺 | P0-3 路由跑/砍；AWQ 实现或书面取舍；能效删叙事或接硬件 |
| 6 | 全部落地后 | 按"语义发现 + 双重修复"新叙事重写 `paper/outline.md`（当前版本停在 10-06 旧故事，§3.2 的 STE 可学习描述与实际 grid 实现也已不符） |

**底线**：outline.md 重写前不要动笔正文；E3 出结果前不要锁定任何表格数字。

---

*证据索引：results/acc_mmlu_sim_g64{.full.json,.records.*.json}（g64 四配置终值）+ acc_mmlu_sim_g64_nacv2/awq.*（v2/AWQ 终值+records）；results/acc_gsm8k_sim_g64.{json,full.json,records.*.json}（E2 收官：gptq 0.305）+ acc_gsm8k_sim_g64_nacv2/awq.*（v2 0.150 / AWQ 0.000 截断）；results/awq_probe_ppl16.json（探针甄别：fold 10.627 / awq_full 11.226）；results/ppl_g64_64seq{,_nacv2}.json（PPL 64-seq 论文口径）；results/acc_mmlu_sim_g128.full.json（g128 三配置）；results/acc_gsm8k_qwen1.5b_g64nac_zs.json（板端 0.005）；results/acc_mmlu_qwen1.5b_g64nac_zs.json（板端 0.5614）；results/ppl_coherence.log / val_nac_g64.json（PPL 16-seq）；results/perf_qwen1.5b_w4a16_full.json（性能）；docs/BOARD_NOTES.md（板端工程记录）。*
