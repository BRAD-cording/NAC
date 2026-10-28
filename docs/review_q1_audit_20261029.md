# NAC 论文全局审计报告（2026-10-29）

> 审计框架：academic-writing 模式 2（5 视角同行评审）+ research-guide 证据纪律（逐条溯源、对抗性验证）+ improve-codebase-architecture 代码扫描 + 10-25 审计基线增量复核。
> 审计对象：paper/outline.md、docs/ 全部、results/ 关键落盘、src/ + scripts/ 核心代码、git 历史（484b66d…d29becc）。
> 基线：[review_publication_20261017.md](review_publication_20261017.md)、[10-25 审计](../subprojects/thesis_audit_2026_10_25/outputs/audit_report.md)、outline §14 决策记录。

---

## 一、实验完成度盘点（对照 outline 承诺）

### 1.1 已收官 ✅

| 板块 | 状态 | 证据 |
|---|---|---|
| sim 轴五方法全矩阵（Table 2） | ✅ 10-22 关账 | acc_mmlu/gsm8k_sim_g64*、ppl_g64_64seq{,_nacv2}.json、awq_probe_ppl16.json |
| g128 边界（Table 4） | ✅ 含 M2 补跑 64-seq 板端 PPL（10-27） | ppl_g128_64seq.json：fp16 11.075 / nac 12.164 / gptq 12.503 / minmax 12.544 |
| 甄别实验 E2/E3/E4 | ✅（E3 结论已被 10-23 事故重释为"两次本地构建同崩"） | TECHNICAL_REPORT §3.2-3.3 |
| 统计统一落盘（M1/F3/M6） | ✅ 10-25 后全部补齐 | stats_pairs.json（22 配对，gold 序列断言）、env_snapshot.json、git 仓库 24 commits |
| zp 统计（Fig 2 数据） | ✅ | zp_analysis.json（int_ratio 0.0016 / frac_mean 0.25 / err 10.67%） |
| 系统表征（Table 5） | ✅ | perf_qwen1.5b_w4a16_full.json |
| 板端 bisect R3→R4d | ✅ 判别分全出 | B1（数据）/B3（opt）/B5（元数据）/B6（模型源）/B7（npu_core）/B8（hybrid）/B2-gdq 全部证伪 |

### 1.2 进行中 ⏳

- **R4-e（grq，B2 剩余主形态）**：00:04 起跑（PID 88944），自动化 922bcde8 看护，预计 ~12:30 出模、~14:00 判别分。这是 B2 整族的最后判别点。
- **recon2**：grq vs vendor 字节相似度（等 R4-e 出模后同法测量）。已有 recon：R3-vendor 41.1% >> R4d-vendor 26.5%（随机 ~6.25%）→ vendor 更接近 normal 而非 gdq。

### 1.3 未完成清单（按阻断性排序）

| # | 项 | 阻断性 | 说明 |
|---|---|---|---|
| 1 | **F1 板端叙事链闭环** | P0 | R4-e 判别 → 归因落定 → R5（NAC-v2 int-zp 重注入板端复评）→ §6.1/6.2/C2 重写。若 grq 仍崩：B2 整族出清，转 target_platform/B4 + 二相归因 |
| 2 | **F2 板端协议统一** | P0 | 基线 GSM8K 0.5550 为 512-token 协议产物，R3/R4 系列为 32-token；统一 32-token 重跑厂商基线。**代码确认 accuracy.py 结果 JSON 只落 fewshot_num 不落 max_new_tokens（L113-147）**——修复落盘字段后重跑 |
| 3 | **二相行为归因**（新增） | P0' | 所有本地构建 MMLU 存活（0.4974-0.5387）/ GSM8K 崩（0.00-0.025）——"MMLU 短生成存活、GSM8K 长链崩"本身待归因；若 R4-e 后仍无解释，Table 1 无法重建，须写成显式 limitation |
| 4 | P1 sim 消融（v1+int-zp，分离 zp 舍入 vs Hessian） | P1 | outline §10 承诺，未跑（成本低：~9min 校准+评测） |
| 5 | LongBench 4K / 校准集消融 | P2 | 可选项，未跑 |
| 6 | 正文写作 | P3 | 正文零字；M4 待复核标注、M5 三线补引、Fig 1/2/4/6 待绘、Table 0 待制 |

---

## 二、缺陷清单

### 2.1 自 10-25 审计以来已修复 ✅

F3（git 仓库，484b66d 起）、M1（stats_pairs.json 22 配对统一重算）、M6（env_snapshot.json 版本+md5 固化）、M2（g128 PPL 64-seq 补跑落盘）、M3（few-shot 降格质性观察，决策拍板）。

### 2.2 仍开放（代码扫描复核确认）

| 编号 | 缺陷 | 位置 | 状态 |
|---|---|---|---|
| F2 | 板端基线与崩塌配置协议不一致；评测 JSON 缺 max_new_tokens 字段 | src/eval/accuracy.py L113-147 | 确认，随 F2 重跑一并修 |
| m1 | extract_number 无 "The answer is" fallback；AWQ 0.000 存在抽取器伪影质疑面 | accuracy.py L77-110 | 确认仍在。探针已裁定真实行为，但**低成本高价值修复**：加 fallback 重抽 AWQ/v1 records（~15min）可消掉该攻击面 |
| m5 | hessian.py "layer-parallel" 实为顺序 for 循环 | src/nac/hessian.py L58-116 | 正文措辞约束仍有效（写"可并行设计"或按实测串行写） |
| m4 | BOARD_NOTES "ratio_mean 0.4" 为修复前旧值 | BOARD_NOTES L63 | 待加注记（meta.json 实为 0.9543） |
| m2 | 死代码（hetero_router.py、20+ probe_*.py） | src/ scripts/ | 开源前归档 |
| M4/M5/m3/p1 | 待复核标注 / 相关工作三线 / outline §9 状态 / outline 新旧叙事并存 | docs/ paper/ | 写作期（R5 一并） |

### 2.3 新增风险（本轮审计识别）

- **N1 bisect 多重比较**：R4 系列已 5 轮单变量对照，若 R4-e 仍崩，剩余假设（target_platform、B4）+ 未建模交互项会让"逐参数 bisect"策略收效递减。建议论文给 bisect 决策树（附录），把"未归因"写成 limitation 而非掩盖——这本身是诚实系统评测的一部分。
- **N2 字节相似度解释边界**：41.1% vs 26.5% 只能支持"vendor 产物更像 normal 输出"的**方向性**判断，不能反推 vendor 精确参数（相似度受打包布局/数值分布影响）。正文引用时必须限定措辞。
- **N3 判别分噪声**：0.000-0.025 区间差异在 n=200 下无统计功效，bisect 全部结论实际依赖"崩 vs 不崩"二值行为判据而非分数差——正文必须声明这一点，否则审稿人会质疑 0.015 vs 0.025 的"变化"。
- **N4 F2 连带**：若 32-token 重跑基线低于 0.555，Table 1 全部配对统计（含 p=7e-25）重算——重跑前任何板端数字不得锁定。

---

## 三、创新性 / 方法论 / 全局性 vs Q2+~Q1

### 3.1 创新性（主编 + 领域视角）

- **C1 系统评测**：first-claim 措辞已限定（"first systematic, controlled three-axis evaluation on RK3576"），M5 三线引用（LLM-KICK/BOSS、Ascend 2602.17693、GPTAQ/LoaQ/BASE-Q）补齐后成立。**Q2 级充分**。
- **C2 语义发现**：⚠️ 原叙事（float-zp 重解释 ~10.7% + E3 对照）已被 10-23 事故推翻，现悬空。候选新叙事："注入格式/语义契约 + 构建路径敏感性 + toolkit 隐藏量化算法族（normal/gdq/grq）对板端可用性的一票否决"（r4_bisect_plan §4 已预留）。**C2 最终强度完全取决于 R4-e 归因成败——这是全文最大的单一不确定性**。若归因失败，C2 降格为质性系统发现，Q1 变难。
- **C3 方法学发现**（PPL/首 token 无法预测长程生成，NAC-v1+AWQ 双方法族复现 + g128 反转边界）：**目前最坚固、最可辩护的贡献**。sim 轴证据链完整闭环，统计全落盘，失败模式分类学（数值错乱型 vs 终止失效型）有原创性。**Q1 级亮点**。
- **C4 部分修复 + 边界刻画**：诚实、可防守（MMLU 追平 p=0.716、GSM8K 0.150 vs 0.305 p=1.2e-4、PPL 代价 +0.6%）。

**创新性评级：Q2 稳（现有 sim 轴证据 + 诚实叙事）；Q1 有条件**——条件 = 板端链闭环后 C2 升级 + M5 引用到位 + 单平台限制坦诚声明。

### 3.2 方法论（方法学审稿人视角）

强项：三轴评测纪律（建议按 p2 扩四轴：+构建路径）、McNemar(χ² 连续性修正)+bootstrap CI 全配对落盘且 gold 序列断言（compute_stats_pairs.py L81-107）、env_snapshot 版本固化、E4 逐位复现验证、校准种子固定（torch.manual_seed(42)）、量化语义严格对齐 NPU 公式 scale*(q-zp)（quantize.py L73-86）。

审稿人必打点（全部需在 Limitations 逐条防守）：
1. 单平台（仅 RK3576）单模型族（NAC 仅 1.5B）——结构性，Q1（TMLR 除外）大概率要求第二平台/模型族；
2. F2 协议混列未闭合（重跑前不可投）；
3. 3B 仅预转基线无量化对照（已定位为扩展数据，可防守）；
4. 校准集敏感性无消融（P2）；
5. AWQ PPL 16-seq 探针口径与主表 64-seq 混列（表注已声明，可防守）。

### 3.3 全局性 / 叙事一致性（Devil's Advocate 视角）

- **最大风险：outline 是"10-23 撤回前后"混合体**。§0.2 摘要发现 1、§6.1/6.2、C2 仍写已撤回的 zp 重解释叙事；§0.1 标题判据仍指向旧叙事；R3-R4d 全部新证据（bisect 七连证伪、二相行为、字节侦察）未入 outline 任何部分。**outline 需要 R5 一次整体重写，不是修补**——p1 比 10-25 时更严重。
- **注入设计的存在性问题**：若本地构建永远无法复现 vendor 基线，"板端校准对照注入"失去对照锚点——论文必须回答"在本地构建崩坏的背景下，extra_qparams 注入链路还能证明什么"。预案：把注入生效判据（R1/R2 的 md5 变化+加载计数）本身作为可复现贡献写入。
- 防御资产已内置：不作 SOTA 主张、统计平手写法、6 条 limitations、写作自查清单——质量高。

### 3.4 收录概率更新（对标 10-17 概率表）

| 情形 | 判断 |
|---|---|
| 今天原样投出 | 同 10-25 结论：大修（板端链悬空 + F2 + outline 新旧并存） |
| R4-e 归因成功 + F2 重跑 + R5 重写 + M5 落地 | **TMLR 小修区间可达**（~60-70%）；APIN/IEEE Access 更稳 |
| R4-e 仍无法归因 | C2 降格为"构建路径敏感性 + 协议契约"质性叙事，仍可投 TMLR 但故事弱一档（~40-50%）；Q1 不可行 |
| Q1（TCAD/TPDS/MLSys） | 需 C2 机制级闭环 + 第二平台/模型族，当前证据域不足——维持 10-17 判断 |

---

## 四、决策建议（优先级）

1. **等 R4-e 判别分（~14:00）**：不改实验计划，自动化在看护。
2. **R4-e 判别后立即**：(a) recon2 字节相似度；(b) F2 协议统一重跑（32-token 基线 + accuracy.py 补 max_new_tokens 字段落盘）——与 R5 构建并行。
3. **m1 修复现在就可做**（不依赖 R4-e）：extract_number 加 fallback → 重抽 AWQ/v1/v2 records → 0.000 判定复核。15 分钟消掉一个审稿人攻击面。
4. **outline R5 整体重写**等 R4-e 归因落定后一次完成（避免第三次返工）；M5 三线引用可先落（不受 R4-e 阻塞）。
5. **投稿定位**：维持 §14 拍板 TMLR；若 R4-e 归因成功且 recon2 支持，可评估 MLSys 系统向。

---

## 五、m1 复核执行结果（2026-10-29 追记）

执行 §四-3 行动项：[recheck_extractor.py](../scripts/recheck_extractor.py) 对落盘 records 用三规则重算（A=现行取最后数字 / Bf=answer-is 首个匹配 / Bl=末个匹配），产物 [extractor_recheck.json](../results/extractor_recheck.json)。

| 配置 | n | A 重算 | Bf | Bl | A 与存档一致率 | B 改变判定数 |
|---|---|---|---|---|---|---|
| sim AWQ | 64 | 0.0469 | **0.3594** | 0.3281 | 0.953 | 22 |
| sim NAC-v1 | 200 | 0.0950 | **0.3000** | 0.2950 | 0.875 | 43 |
| sim minmax | 200 | 0.0350 | **0.3100** | 0.2500 | 0.910 | 56 |
| sim GPTQ | 200 | 0.2800 | **0.3850** | 0.3800 | 0.825 | 21 |
| sim NAC-v2 | 200 | 0.1300 | **0.3400** | 0.3250 | 0.850 | 46 |
| 板端 vendor zs | 200 | 0.0200 | 0.0200 | 0.0200 | 0.455¹ | 0 |

¹ 板端 records gen 截断 200 字符而原始协议 512-token——重算不可用（预期内，佐证 F2）。

**发现（重大，需用户拍板）**：
1. 抽取规则是**全配置一阶混杂**，非 AWQ 特有：answer-is 优先规则下五方法 GSM8K 全部大幅上移，**E2 的 gptq 4× 分化（0.305 vs 0.07）在 Bf 规则下压缩为 0.385 vs 0.30-0.36**——"问题 A 幅度"与"终止行为 × 抽取规则"存在混淆。
2. 板端证据不受影响：NAC-v1 板端 GSM8K 0.005 的失败模式（预测全小数、长度中位 3）是数值损坏的直接证据，与抽取规则无关；C3 的"PPL-长程脱钩"方向仍成立但幅度需重估（AWQ 若实为 ~0.36，"0.000 灾难"表述必须撤换）。
3. **截断混杂限制**：records.gen 仅存 300 字符（sim）/200（板端），原始评分用推理全文——本复核只能定性（规则敏感 + 秩压缩），不能出论文级数字。Bf=0.3594 对 AWQ 是下界估计。
4. 处置选项（待拍板）：
   - **A 线（论文级，~2-3 天 sim 机时）**：eval_gsm8k_sim.py 加 dual-metric（last-num + answer-is）落盘全文（或前 1000 字符）+ 协议字段，重跑 5 配置 sim GSM8K；
   - **B 线（披露级，零机时）**：Table 2 保持现行数字，正文披露抽取规则 + 引用本复核为敏感性分析（注明截断限制）；
   - 任一线上 m1 的"加 fallback 改默认抽取器"均**暂缓**——R4-e 评测在即，改默认会破坏可比性。

## 审计覆盖声明

- 通读：outline 全文、review_publication、10-25 审计、TECHNICAL_REPORT §4-5 日志、git log 25 commits。
- 复核：BOARD_NOTES（R3-R4e 段）、results/ 关键 JSON 存在性（stats_pairs/env_snapshot/zp_analysis/ppl_g128_64seq 均在）、src/eval + scripts 统计链代码扫描（Explore 子代理）。
- 未覆盖：R4-e 产物（构建中）；results/ 全量 239 文件仅抽样；本轮未重算 stats_pairs.json 数值（信任 10-25 Agent A+B 对账 + dbfcb50 commit）。
