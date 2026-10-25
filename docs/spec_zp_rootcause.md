# Spec: NAC 板端 GSM8K 灾难崩溃根因甄别与修复

日期: 2026-10-16
状态: 已批准执行（2026-10-16 当日批准；E2/E3/E4 已编排进 watcher 自动链。10-18 注记：sim 层发现 NAC 校准目标本身缺陷（问题 A，与 NPU 无关），与本题的 NPU float-zp 重解释嫌疑（问题 B）构成双问题叠加；A 修复=NAC-v2 已实施，B 由 E3 板端对照裁决，详见 docs/tasks_zp_rootcause.md 进度注记）
关联: 10-15 版评审 P0-1（review_q2_20261015.md 已删除；路径 C 已打通，进入注入语义正确性甄别）

## 1. 背景与已确立事实

| 实验 | 结果 | 状态 |
|---|---|---|
| 板端 NAC GSM8K zs (n=200) | **0.0050** vs 基线 0.5550（1/200 vs 111/200，McNemar p=7e-25） | 灾难崩溃已坐实（records: acc_gsm8k_qwen1.5b_g64nac_zs.records.json，2026-10-16 11:20 落盘） |
| 板端 NAC MMLU zs (n=969) | 0.5614 ≈ 基线 0.5583（p=0.878 平手） | 已完成 |
| sim MMLU | NAC 0.3932 vs minmax 0.4675（p<1e-4 劣化）；PPL NAC 11.408 < minmax 11.738（更优） | 已完成 |
| 板端预测格式 | '52.0' 类小数格式为通用生成行为（基线同样输出），records `ok` 字段为官方评分 | 非解析 bug |
| g64 qparams 时效 | results/nac_qwen1.5b_g64.qparams.pt（Oct 8 12:45）即最终版，val_nac_g64_fix.log（18:22）复核 PPL 一致 | 无过期问题，疑云关闭 |
| qparams 结构 | `{int层号: {模块名: {scale fp32(1536,24), zp fp32(1536,24), ratio(仅NAC)}}}` | 已探明 |
| ppl_coherence | 进程再次死亡（log 停 3/16，10:54），与 sim 评测（6.7GB）内存竞争 | 待内存窗口 |

核心矛盾：板端 MMLU 平手但 GSM8K 崩溃；sim MMLU 劣化但 PPL 更优。

## 2. 假设

- **H1（头号）**: NPU runtime 对 W4A16_G64 的 float zp 做取整/定点重解释，NAC zp 大量非整数 → 等效权重误差 scale×|frac(zp)|。GSM8K 需精确算术推理 → 崩；MMLU 四选一容错高 → 平手。sim 栈按 float zp 伪量化 → PPL 更优但 generative 劣化。
- **H2**: toolkit 读取 extra_qparams 但对 float zp 应用存在语义偏差（与 H1 同为「板端特有」大类）。
- **H3**: NAC qparams 语义本身低质（float zp 软连续在数学推理上天然劣）——预测 sim GSM8K 上 NAC 也崩。

## 3. 甄别实验（判别矩阵）

| 实验 | 成本 | 判别逻辑 |
|---|---|---|
| **E1 zp 分布分析**（零内存，立即可做） | ~1min CPU | NAC-g64 vs GPTQ-g64 的 zp 小数部分分布。若 NAC zp 广泛非整数而 GPTQ 全整数 → 量化偏差 = scale×|frac|，对照权重量级评估影响；GPTQ 作为「板端正常」对照 |
| **E2 GSM8K sim×3**（队列自动跑，零干预） | 排队中 | fp16 完成后队列自动执行 minmax/nac/gptq。NAC 崩 + minmax 正常 → H3 坐实（qparams 本身）；全部正常 → 板端特有（H1/H2） |
| **E3 minmax 对照板端构建**（需 ~7GB 内存窗口） | 构建 ~1h + 板评 | 注入自实现 minmax qparams（与 NAC 同为 float zp 语义）上板评测 GSM8K zs。对照也崩 → float-zp 注入路径普遍失真（H2 变体）；对照正常 → NAC 特有（H1 或 H3 板端侧） |
| **E4 ppl_coherence 重跑**（需内存窗口） | ~35min | 当前代码 × 当前 qparams 复算 16seq PPL，对照 10-08 的 fp16 10.627/minmax 11.738/nac 11.408——排除代码改动引入的 qparams 应用回归 |

## 4. 修复路径（由甄别结果决定）

- **H1/H2 坐实** → src/nac/quantize.py 增加 integer-zp 约束（zp 限整数格点，重校准）→ 重生成 qparams → 重建注入产物 → 板端复评 GSM8K zs（达标：≥0.50，接近基线 0.555）
- **H3 坐实** → 修 NAC 校准质量（审查 ratio 加权与 grid 搜索粒度），sim GSM8K 作为快速迭代回路
- 无论何种结果，板端论文表格须注明 zp 语义（P0-2 量化器轴区分）

## 5. 资源与排程约束

- sim fp16 评测进行中（416/969，~43s/题，ETA 10-16 ~18:30），期间 avail 内存 ~7GB，**禁止**并行启动构建/PPL 类重任务（已有两次 OOM 级教训）
- 队列顺序：fp16 → GSM8K sim×3（~7h）→ GPTQ g128 校准 → MMLU g128×3（~10h）
- E3 构建窗口：GSM8K sim×3 结束后（约 10-17 凌晨）插入，或人工决策牺牲 MMLU g128 段
- E4 与 E3 不可并行，同窗口二选一

## 6. 验证标准

- E1 产出量化结论（整数比例、|frac| 均值、等效权重误差相对幅度）
- E2/E3 完成后填写判别矩阵，唯一根因落定
- 修复产物板端 GSM8K zs ≥ 0.50 且 MMLU zs 不低于 0.55
