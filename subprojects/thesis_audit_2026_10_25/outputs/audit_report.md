# NAC 论文项目 Q1/Q2 投稿就绪度全局审计报告（2026-10-25）

> 审计框架：thesis-audit-reviewer 工作流；覆盖 paper/、docs/、src/、scripts/、results/ 全量。
> 审计基准：Q2+/Q1 首次投稿不大修。每条发现含位置、证据、修改要求、优先级、确定性。

## 总评

**sim 轴证据链坚固（对账全过），板端轴因 10-23 事故整体悬空（R3 构建中），统计落盘与可复现性包存在系统性缺口。当前直接投稿仍会大修；R3-R5 闭合 + 下列 M 级修复后可达"小修"区间。**

- 数据对账：outline 声称的 sim 轴全部数字（五方法矩阵、g128、PPL 两口径、meta 耗时、zp 统计、perf、3B zs）与 results/*.json **逐项一致** ✅
- 板端轴：Table 1 全部三行依赖"注入生效"叙事，已被 10-23 事故撤回；等待 R3/R4 重建
- novelty：first-claim 低风险（需限定措辞）；"zero-point 重解释机制级因果证明"在商品边缘 NPU LLM 栈上无直接前人（Ascend 案例 arXiv 2602.17693 只报现象未做机制归因，可引为独立佐证）

## 致命（阻断投稿）

### F1 板端证据链整体悬空
- 位置：paper/outline.md §0.2/§6.1/§6.2/C2；docs/TECHNICAL_REPORT.md L112-152
- 证据：outline 顶部 10-23 更正（三次注入产物 md5 相同、`0 extra qparams loaded`）；`results/acc_mmlu_qwen1.5b_g64_nacv2intzp_zs.json` 的 model 字段为 `models/nacv2_qwen1.5b_g64_intzp.rkllm`（10-22 失效 P0 构建产物）
- 影响：C2"语义发现"、E3 对照逻辑（"minmax float-zp 注入同崩→非 NAC 特有"实际是两次本地普通构建同崩，未对照任何变量）、摘要发现 1、标题 R2 选项全部不可用
- 修改要求：R3（构建中）→ R4（接口归一对照：新格式烘焙 minmax vs NAC-v2 同接口）→ R5 回填；板端旧 JSON 一律不入正文；C2 按 grill-me 决策 5 改写为"注入格式/语义契约"层
- 确定性：确认问题

### F2 板端 GSM8K 基线协议疑不一致
- 位置：docs/BOARD_NOTES.md L74（"GSM8K 用 --max-new-tokens 512（32 会截断）"）vs outline §5 协议矩阵（板端 GSM8K-Short 32 token 直答）
- 证据：`results/acc_gsm8k_qwen1.5b_w4a16_zs.json` 无 max_new_tokens 字段；`src/eval/accuracy.py` L118 默认 32。基线 0.5550 为 10-07 512-token 协议产物，崩塌配置（0.005）为 32-token 协议
- 影响：Table 1 的 McNemar p=7e-25 若跨协议配对则为方法论硬伤
- 修改要求：R3/R4 重评时统一 32-token 协议并重跑基线；评测 JSON 增加协议字段落盘
- 确定性：疑似问题（基线可能未重跑）

### F3 无版本控制
- 证据：git status 显示 nac_paper/ 为 rknn-llm 仓库 untracked 目录，无 git history
- 影响：代码开源声明（outline §10 P3）无可快照基础；审稿期无法引用 commit hash
- 修改要求：投稿前 git init + 首次 commit + tag；results/*.json 一并入库
- 确定性：确认问题

## 大修

### M1 统计检验未统一落盘
- 证据：McNemar b/c/p 值散落于 docs 补记文本（review_publication L7-11 等）；Glob 无 stats/mcnemar 独立文件；outline §9 Table 3 自注"部分已算，其余需补算统一表"
- 修改要求：写统一脚本重算全部配对（含 continuity correction + bootstrap CI95），落盘 `results/stats_pairs.json`；Table 3 由脚本直出
- 位置：scripts/eval_gsm8k_sim.py L157-176（配对按 `zip(mm[:n], nc[:n])` 位置对齐，依赖两边样本顺序一致——需在脚本中断言题号对齐）

### M2 g128 PPL 口径混列
- 证据：`results/val_nac_g128.json` num_seqs=16（16-seq 探针口径），outline Table 4 无口径标注，与 Table 2 的 64-seq 主表同文呈现（g128 minmax 11.988 ≠ g64 64-seq 12.225，审稿人必问）；gptq g128 PPL 缺失
- 修改要求：Table 4 表注声明 16-seq 口径（三轴纪律自己的条款），或补 64-seq g128 PPL；gptq g128 PPL 补跑或注明"—（未测）"

### M3 few-shot 协议轴发现无落盘
- 证据：1.5B 5-shot 0.4469、3B 5-shot 0.10 仅存在于 BOARD_NOTES.md L51-58 文本；3B 为 n=50 子集；无 JSON/records
- 修改要求：二选一——(a) 重跑落盘完整 969 题 5-shot JSON；(b) 降格为质性观察写进 §6.7，3B n=50 标注为初步观察。不许以现状态直接入正文表格

### M4 旧板端结论未打"待复核"
- 位置：TECHNICAL_REPORT.md L112-113（精度矩阵 E3 行）、L131-138（H1 坐实/H2 排除）、L146-152（E2/E3/E4 表 E3 行"收官/问题 B 坐实"）
- 证据：outline §13 R5 明确要求标注；顶部警告存在但正文未逐处标记，快速浏览仍会误读
- 修改要求：R5 执行时逐处加"⚠️ 10-23 事故后待复核/已撤回"行内标记；H2"已排除"必须反转改写（R1 证明 H2 恰恰为真）

### M5 相关工作覆盖缺口（Agent D 网络调研结论）
- 必补引用三线：
  1. 量化评测方法学线（C3 的合法性来源）：LLM-KICK（ICLR 2024）、BOSS（2406.12928）、QSpec（2410.11305）、LongGenBench（2410.04199）、Illusion of Equivalency（2607.08734）
  2. NPU 部署 gap 线（审稿人必查）：**Ascend PTQ 案例（2602.17693，最接近先例，必引并写差异：其未做 zp 机制归因与因果对照）**、Is INT8 Portable（2609.16085）、MixLLM（2412.14590）、TileFuse（2606.11357）
  3. 2024-26 PTQ 对表：GPTAQ（2507.11017）、LoaQ（2509.06297）、BASE-Q（2506.15689）等
- first-claim 措辞：改为 "first systematic, controlled three-axis evaluation on RK3576（学术、可复现）"，引用 rknn-llm repo 与 RKNN3 SDK 公告划清灰色文献边界
- venue 参考：TMLR（方法学友好）/ MLSys / TCAD/TPDS（系统向 Q1）；原计划 APIN(Q2)/IEEE Access 仍是稳妥档

### M6 可复现性包
- 证据：Agent B 代码审计——toolkit/transformers/torch 版本未入任何 meta.json；校准数据生成已固定（wikitext-2 validation + random.Random(seed)）；种子 42 已设
- 修改要求：所有 calibrate/eval 脚本 meta 落盘软件版本 + 构建命令 hash；写 requirements.txt / 环境快照；开源声明中给出 commit tag

## 小修

- m1 `src/eval/accuracy.py` extract_number 无 "The answer is" fallback（L77-110）：AWQ 0.000 的判定依赖"取最后数字"规则；正文须写明抽取规则 + none_rate=0 + records 质性证据链（"The answer is 83" 但不终止），否则审稿人会质疑 0.000 是抽取器伪影
- m2 死代码处置：src/routing/hetero_router.py（已砍功能）、20+ probe_*.py——开源时移 scripts/.archive/ 或删除
- m3 outline §9 状态过时：Fig 2 zp 数据已落盘（zp_analysis.json：int_ratio 0.001584/frac_mean 0.25/err 10.67% 与声称一致；mmctrl_zp_stats.json 亦在），标注仍写"需补算"——改为"数据齐"
- m4 BOARD_NOTES.md L63 "ratio_mean 稳定在 0.4" 为 10-07 修复前旧值（meta.json 实为 0.9543），需加注记防误引
- m5 NAC-v2 hessian.py 的"块内 layer-parallel"实为顺序 for 循环（L58-116）——正文不得声称并行实现，或改为"可并行设计"

## 润色

- p1 outline 内部自洽：顶部 10-23 撤回声明与 §6.2 旧论证（"判定 H1 成立"）并存——写作时整体重写 §6.2
- p2 三轴纪律扩为四轴（量化器来源 × 执行栈 × 评测协议 × **构建路径**（本地 qemu 构建 vs 厂商预转））——把事故转化为方法学贡献点；Table 1 加构建路径列
- p3 3B 定位声明（扩展数据非方法证据）已有，保持

## R3 接续预案（守望中，shell b61e8ddb / r3-x86-wd 在岗）

- ≈0.55：g64×NAC 误差坐实 → R2.3 数值一致性抽检 → R4 对照 → C2 按预案改写
- ≈0.005：本地构建配置问题 → 离线 diff（本地 plain 产物 vs 厂商基线 qparams）→ bisect（dataset/optimization_level/num_npu_core/toolkit 版本）

## 审计分母覆盖记录

| 对象 | 状态 |
|---|---|
| paper/outline.md 全文 | 已审（含 §13 事故链） |
| docs/ 5 文件 | 已审（TECHNICAL_REPORT 关键区间、BOARD_NOTES 全文、review_publication 全文） |
| src/nac + eval 核心逻辑 | Agent B 已审（分级结论见 M6/m1/m5） |
| scripts 核心 9 个 | Agent B 已审；probe_* 归档建议见 m2 |
| results/*.json 82 个 | 抽样对账 20+ 关键文件（Agent A + 本审计直读）；数字对账全过 |
| 相关工作 | Agent D 网络调研 29 篇（M5） |
| 统计实现 | Agent B 审（M1） |
| blocked | R3 产物（构建中，无法审）；hotl:code-review 不适用（项目无 git diff 基线） |
