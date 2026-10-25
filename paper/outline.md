# 论文骨架 v2.0（2026-10-22 全实验矩阵收官版）

> ⛔ **2026-10-23 重大更正（写作前必读）**：P0 板端复评暴露**注入链路从未生效**——三次"注入"构建（NAC-v1 float-zp、minmax float-zp/E3、NAC-v2 int-zp/P0）产物 **md5 完全相同**（`3bd8c0ba73f056a98646c03959e80eec`），工具链日志均为 `Using extra qparams: …` 后接 **`0 extra qparams already found and loaded!`**。
> 结论：**板端从未跑过任何 NAC/AWQ 权重**；§6.1、§6.2、E3 对照、问题 B（zp 重解释）判定、贡献 C2 全部**撤回，暂停写作**。板端 0.5614/0.005 是"toolkit 本地普通构建"的表现，且该构建与厂商预转基线（0.555）差异巨大——这是**待查的第二个独立问题**（疑似 g64 或构建配置），须先解决才能做任何校准对照。
> 恢复路径见文末 §13。**在 §13 全部验证通过前，摘要/引言/§6.2 不得引用任何板端 CALIBRATION 结论。**

> **取代关系**：本文件整体替换 10-06 版骨架。旧版与实际不符处全部作废：STE 可学习 scale/zp（实际为 grid 截断搜索）、异构路由小节（10-17 已砍，零数据）、7B 模型（未跑）、tokens/J 能效（无电流采样，不可测）、"NAC 恢复 1–3%"（实际为部分修复 + 边界刻画）。
> **数字口径**：所有数值以 `results/*.json` 终值为唯一来源，本文件内每个数字后标注来源文件；正文动笔前须回查一遍。
> **叙事已定判（10-22）**：系统评测 + 语义发现 + 部分修复 + 边界刻画。不做 SOTA 主张。

---

## 0. 标题与摘要

### 0.1 标题（二选一，按板端复评结果触发）

- **R1（推荐，当前证据即成立，系统评测向）**
  *Zero-Point Reinterpretation in W4A16 Post-Training Quantization: A System-Level Study on Rockchip RK3576*
- **R2（仅在 §7 必补实验 P0 达标后启用，双修叙事）**
  *NPU-Faithful Zero-Point Semantics for W4A16 LLM Quantization on Rockchip RK3576*

**判据**：R2 要求板端 int-zp 注入后 GSM8K 显著高于 float-zp 的 0.005 且达到"可用"水平。未复评前不得用 "NPU-Aware"、"Hardware-Faithful" 等形容词做标题——评审原文要求"投稿前题目里每个形容词必须有数据背书"。

### 0.2 摘要骨架（~250 词，五句式）

1. 现状：边缘 SoC（RK3576）以 W4A16 per-group 非对称量化部署 LLM，但校准发生在 GPU/x86 侧，校准语义与 NPU 执行语义的差异未被系统刻画。
2. 方法：建立三轴评测纪律（量化器来源 × 执行栈 × 评测协议），在 RK3576 真机与 x86 fp32 伪量化双栈上对照五种校准族（minmax / GPTQ / AWQ / NAC-v1 / NAC-v2）。
3. 发现 1（语义）：NPU 对非整数 zero-point 的重解释使等效权重误差达 ~10.7%，板端 GSM8K 从 0.555 崩至 0.005；以 minmax float-zp 对照注入（0.005）坐实其非 NAC 特有。
4. 发现 2（方法学）：困惑度与首 token 精度无法预测长程生成能力——激活加权目标族（NAC-v1、AWQ）在 PPL/MMLU 轴无损甚至占优，却在 GSM8K 崩溃；整数 zp + Hessian 补偿修复使 MMLU 追平 GPTQ（0.4469 vs 0.4520，p=0.716），GSM8K 部分恢复（0.070→0.150）但仍显著落后（p=1.2e-4）。
5. 结论：给出边界条件（组粒度 g64/g128 反转、任务类型）与对 NPU 工具链的可操作建议（整值 zp 或显式声明反量化语义）。

---

## 1. Introduction（~1100 词）

- **动机（3 段）**
  1. 边缘 LLM 落地依赖 INT4 权重量化；RK3576 类 SoC 的 W4A16 流水线已成事实标准，但校准工具链（rknn-llm/GPU 侧）与实际 NPU 反量化路径之间存在未验证的语义契约。
  2. 既有 PTQ 文献（GPTQ/AWQ/OmniQuant/SmoothQuant）以 FP16 模拟量化误差作为优化目标，默认"模拟 = 部署"；本文证明该假设在 NPU 上不成立。
  3. 由此带来的实践后果：一个 PPL 与 MMLU 都正常的 4-bit 模型，可能在算术推理上归零——评测协议选择本身是可靠性的组成部分。
- **问题陈述**：三个可证伪问题
  - Q1（语义）：NPU 执行的非对称反量化与校准侧 float-zp 语义是否一致？
  - Q2（归因）：若不一致，板端精度崩溃应归因于硬件语义、校准目标，还是二者叠加？
  - Q3（修复与边界）：修复后能达到什么水平，失效边界在哪里？
- **贡献（4 条，每条一句话 + 一句证据）**
  - C1 **系统评测**：首个 RK3576 W4A16 LLM 量化系统评测（板端 1.5B/3B，MMLU 969 / GSM8K 200，零样本协议），含注入链路（extra_qparams）与构建病理的可复现记录。
  - C2 **语义发现**：定量给出 float-zp 在 NPU 上的重解释代价（等效权重误差 ~10.7%）并以对照实验证伪"NAC 特有"解释（E3：minmax float-zp 板端 GSM8K 0.005 同崩、MMLU 0.5614 平手）。
  - C3 **方法学发现**：困惑度与首 token 精度不是长程生成的代理指标；该脱钩在两个独立的激活加权方法族（NAC-v1、AWQ）上复现，属方法族现象而非实现缺陷（AWQ fold 等价性 10.627 精确复现 fp16、PPL 11.226 三方法最优，但 GSM8K 0.000）。
  - C4 **部分修复与边界刻画**：整数 zp + Hessian 补偿使 MMLU 追平 GPTQ（p=0.716）、GSM8K 提升 2.1×（p=0.012）；但差距未消除（p=1.2e-4），并给出两组边界（组粒度 g128 反转 +6.71pp；任务轴短上下文/长程）。
- **不作的主张（明写进 Introduction 末段，预防审稿人误读）**：不主张 SOTA；不主张 NAC 全面优于 GPTQ；不主张板端修复已完成（见 §7 P0）。

---

## 2. Related Work（~900 词）

- **2.1 PTQ 与校准目标**：GPTQ（Hessian 误差补偿、列序 Cholesky）、AWQ（激活感知 scale + 等价折叠）、SmoothQuant（激活-权重迁移）、OmniQuant（可学习裁剪）；共同点：优化目标建立在"模拟即部署"假设上。
- **2.2 边缘 NPU 部署栈**：rknn-llm / RKNN Toolkit、MNN、ExecuTorch、TFLite；共同点：量化参数为整数域设计（int8/int4 对称或非对称），非对称 zp 的运行时语义少有公开规范。
- **2.3 量化评测的有效性**：PPL 与下游任务的弱相关性（近年零散报告）；本文将其从"相关性弱"升级为"存在系统性反向排序"，并给出机制解释（长程误差累积 + 终止行为）。
- **定位段（1 段，必须写清）**：本文不是提出更强 PTQ 算法，而是（a）揭示模拟-部署语义契约的破裂，（b）给出可复现的判别实验，（c）刻画修复的能力边界。与 C1–C4 一一对应。

---

## 3. Background and Evaluation Framework（~1200 词，本文方法论核心）

### 3.1 RK3576 W4A16 反量化路径
- 公式 `W_dequant[i] = scale[g] · (W_int4[i] − zp[g])`，uint4 `[0,15]`，per-group（g64/g128）独立 scale/zp（[quantize.py](../src/nac/quantize.py)）。
- **关键歧义**：zp 在公式中为实数域占位符；硬件实现是"整值查表"还是"浮点参与运算"未在公开文档中声明。本文的 H1 即针对此歧义。
- 平台与定频：NanoPi RK3576（4×A53@2.016GHz + 4×A72@2.208GHz，16GB，2 核 NPU@950MHz，DDR@2736MHz）；4×A72 参与推理。

### 3.2 三轴评测纪律（C1 的方法论骨架）
- **轴 1 量化器来源**：toolkit 内置（minmax/GPTQ）vs 外部注入（extra_qparams，NAC/AWQ/fold）。
- **轴 2 执行栈**：板端 NPU 真机 vs x86 fp32 伪量化（`apply_qparams` 逐层写回）。
- **轴 3 评测协议**：任务 × shot × 采样数与生成长度。
- **纪律条款**：任何跨轴比较必须显式标注；同一表格内不得混列不同轴的数值。附反例（历史教训：把 sim 量化器与板端量化器混入同一列导致误判）。

### 3.3 伪量化栈的定位与有效性边界（必须写，防审稿人攻击）
- **绝对量不可比**：sim fp16 MMLU 0.4634 vs 板端 w4a16 基线 0.5583（**9.5pp 栈间差**，[acc_mmlu_sim_g64.full.json](../results/acc_mmlu_sim_g64.full.json) vs BOARD_NOTES）——差异候选：32-token 生成上限、批处理/padding、prompt 模板与 chat template、fp32 与 NPU 数值路径。
- **相对量可用**：E2 判别有效性（GPTQ 0.305 vs minmax/nac 0.07，同协议同批处理）与 E4 复现性（PPL 与 10-08 首测逐位一致）共同支撑"同栈配对比较可信"。
- **明写结论**：本文将 sim 轴定位为**受控相对比较器**，不用于预测板端绝对值；板端结论只由板端数据支撑。

---

## 4. Calibration Methods under Study（~1500 词）

统一记法：目标 `min Σ_g a ⊙ (W − dequant(q(W)))²`，`a` 为激活统计。

- **4.1 minmax（toolkit normal）**：per-group 极值定标，无误差补偿；作为"语义族对照"（float-zp 族代表）。
- **4.2 GPTQ**：Hessian（`percdamp=0.01`）列序补偿 + maxshrink 网格（0.8，100 格），整值 zp；校准 6527s（[gptq_qwen1.5b_g64.meta.json](../results/gptq_qwen1.5b_g64.meta.json)）。
- **4.3 AWQ（自实现）**：激活感知 per-channel scale 搜索（α 网格 21 格，step 0.05）+ clip 网格 + 等价折叠（fold 存 scale 向量 ~1.4MB，`apply_fold` 位级重建）+ 整值 zp；校准 1511s。**折叠等价性作为实现正确性的断言**（见 §6.4）。
- **4.4 NAC-v1**：GPTQ 式 block 顺序传播 + 激活加权 per-group 截断比网格 `κ∈{1.0,…,0.4}`；float-zp；校准 **521s**（最廉价，[nac_qwen1.5b_g64.meta.json](../results/nac_qwen1.5b_g64.meta.json)）。
- **4.5 NAC-v2（本文修复版）**：在 v1 上叠加 (a) 整值 zp、(b) 块内 layer-parallel Hessian 补偿（`act_weight=diag(H)`，`percdamp=0.01`），(k) 保留 κ 网格；校准 4259s（[nacv2_qwen1.5b_g64.meta.json](../results/nacv2_qwen1.5b_g64.meta.json)）。
- **4.6 实现与工程约束（1 段，体现可复现性）**：注入格式（HF 命名 196 层全量）、构建必须输入 fp32 原始模型（dequant 重写模型在 qemu 下 `Optimizing 0/28` 卡死）、qparams 禁落 `/tmp`。三处 AWQ 静默 bug 的教训可压缩为一句"等价性断言先于精度评测"。

---

## 5. Experimental Setup（~900 词）

- **硬件**：RK3576（定频四项：CPU/NPU/DDR），16GB；能效不可测（无电流采样），效率叙事仅限 TTFT/decode/内存三轴。
- **模型**：Qwen2.5-1.5B-Instruct（全矩阵）；Qwen2.5-3B-Instruct（仅板端预转基线，作系统评测扩展 + few-shot 现象的第二数据点）。
- **数据**：MMLU 969 题（1000 采样池去重）、GSM8K 200 题、Wikitext-2 heldout（16×1024 / 64×1024 两口径）。
- **协议矩阵**（表 0，正文首表）：
  | 栈 | 任务 | shot | 生成长度 | 指标 |
  |---|---|---|---|---|
  | 板端 NPU | MMLU | 0 | 32 | acc |
  | 板端 NPU | GSM8K-Short（`Question:…\nAnswer:` 直答） | 0 | **32** | acc |
  | sim fp32 | MMLU | 0 | 32 | acc_gen / acc_ftok |
  | sim fp32 | GSM8K | 2 | 512 | acc |
  | sim fp32 | PPL | — | 1024 | PPL |
- **few-shot 退化的协议声明（重要）**：板端 1.5B MMLU 5-shot 0.4469 < zs 0.5583；3B 5-shot 崩至 0.10（复述示例）。故板端统一 zs；该现象本身作为 §6.7 的系统发现报告，不归因于量化。
- **统计**：McNemar（χ² 连续性修正）、bootstrap CI95、平手判据（p>0.05 且差 < CI 宽度）；配对按数组位置（records `i` 为批计数，非题号）。

---

## 6. Results（~3000 词）

### 6.1 板端基线与其崩塌（Table 1）
| 配置 | MMLU zs | GSM8K zs | 判读 |
|---|---|---|---|
| toolkit w4a16 基线 | 0.5583 | **0.5550** | 参考 |
| NAC-v1 注入（float-zp） | 0.5614（p=0.878 平手） | **0.005**（1/200，p=7e-25） | 灾难 |
| minmax float-zp 对照注入（E3） | 0.5614（平手） | **0.005** | **同崩 → 非 NAC 特有** |
- 论述要点：崩塌的**任务选择性**（MMLU 四选一容错 vs GSM8K 精确算术）；失败模式（预测全为小数、长度中位数 3、无空预测 → 仍在生成）。
- **协议差异必须在正文点明**（防混淆）：板端 GSM8K 是 **32-token 直答**（zs，`Question:…\nAnswer:`），不是 sim 的 2-shot CoT/512——所以板端崩塌**不依赖长程生成**，是数值本身已坏；而 sim 轴 NAC-v1 的失效是"短任务尚可、长程才崩"。两者是**不同层级的失效**（表示损坏 vs 误差累积），§6.2 与 §6.4 须分别归因，不可互相引用为同一证据。
- 证据：[acc_mmlu_qwen1.5b_g64nac_zs.json](../results/acc_mmlu_qwen1.5b_g64nac_zs.json)、[acc_gsm8k_qwen1.5b_g64nac_zs.json](../results/acc_gsm8k_qwen1.5b_g64nac_zs.json)。

### 6.2 语义归因：zp 重解释（C2 的核心论证）
- Fig 2a：NAC zp 分布（99.84% 非整数，|frac| 均值 0.25，范围 [−0.063, 15.105]）vs GPTQ 100% 整值。
- Fig 2b：若 NPU 取整 zp，等效权重误差分布（均值 ~10.7%，最大 ~3.9×）。
- 论证链（写作顺序即逻辑）：zp 统计 → 误差上界 → 板端崩塌量级 → E3 对照排除"NAC 特有"→ 判定 H1 成立。
- 反证排除：H2（toolkit 未应用 extra_qparams）——若未应用则不可能崩、且 MMLU 精确平手亦不可解释。

### 6.3 sim 轴五方法全矩阵（Table 2，论文主表）
| 方法 | MMLU(969) | GSM8K(200) | PPL 64-seq | 校准耗时 |
|---|---|---|---|---|
| fp16 | 0.4634 | — | 11.075 | — |
| minmax | 0.4675 | 0.075 | 12.225 | — |
| GPTQ | 0.4520 | **0.305** | 12.168 | 108.8 min |
| AWQ | 0.4396 | **0.000** | 11.226¹ | 25.2 min |
| NAC-v1 | 0.3932 | 0.070 | **11.893** | 8.7 min |
| **NAC-v2** | **0.4469** | 0.150 | 11.962 | 71.0 min |
¹ AWQ PPL 为 16-seq 探针口径（awq_full），其余为 64-seq；表注必须写清（三轴纪律）。
- 三个必须点出的读数：
  1. **秩反转**：PPL 最优（NAC-v1 11.893）与长程生成最差（0.070）同体；GPTQ 反向（PPL 最差 12.168、GSM8K 最好 0.305）。
  2. **MMLU 判别力饱和**：NAC-v1 在 sim MMLU 劣于 minmax 7.43pp，但板端 MMLU 反而平手（0.5614 vs 0.5583）——板端 MMLU 对两类量化误差均不敏感，不能作为判别任务；这是"为何必须做长程任务"的实证理由。
  3. **成本**：NAC-v1 521s 即可完成 1.5B/28 块校准（边缘 SoC 板载 CPU 可行性的关键卖点），v2 4259s 仍低于 GPTQ 6527s。

### 6.4 问题 A：长程误差累积与"激活加权族"的双例证（C3）
- **E2 三方分化**：GPTQ 0.305 独活 vs minmax 0.075 / NAC 0.070（同协议配对）→ 判据有效，崩溃是真实校准质量问题。
- **AWQ 作为第二族独立证据（本文最强的新增论据）**：
  - 先排除实现缺陷：fold_only PPL **10.627**（精确复现 fp16 10.627）→ 跨块折叠端到端无损；awq_full **11.226** 优于同口径 NAC-v1 11.408 / minmax 11.738。
  - 再指出表征失效：GSM8K 0.000，records 显示 CoT 步骤正确（gold=83 题算出 "The answer is 83"）但**不终止、跨题续写**，抽取错位。
  - 结论：EOS 终止退化是真实校准行为，且该失效在"无损 PPL、无损首 token"的表象下不可见。
- **失败模式分类学（建议做成一个小表）**：NAC-v1/minmax = 数值错乱型（预测变小数）；AWQ = 终止失效型（内容对但不收尾）。二者在 PPL 上都不可见。

### 6.5 修复效果（C4 上半）
- NAC-v2 vs NAC-v1 MMLU：0.4469 vs 0.3932（+5.37pp，b=98/c=46，**p=2.1e-5**）。
- NAC-v2 vs GPTQ MMLU：−0.51pp（b=58/c=63，**p=0.716**，平手）→ **短上下文任务全额恢复**。
- NAC-v2 vs NAC-v1 GSM8K：0.150 vs 0.070（b=26/c=10，**p=0.012**）→ 长程部分恢复。
- NAC-v2 vs GPTQ GSM8K：0.150 vs 0.305（b=15/c=46，**p=1.2e-4**）→ **未追平**。
- PPL 代价：11.962（vs v1 11.893，+0.58%）→ "以近零困惑度代价换取长程能力"。
- **诚实的成分归因声明**：v2 同时改动整值 zp 与 Hessian 补偿，sim 轴无法分离二者贡献（整值 zp 在 sim 忠实语义下不改变语义，仅改变舍入）；板端分离需 §7 P0。此处**不得**写"哪一项贡献多少"。

### 6.6 边界条件一：组粒度（Table 4）
| g128 配置 | MMLU | PPL |
|---|---|---|
| minmax | 0.3622 | 11.988 |
| NAC-v1 | **0.4293**（+6.71pp，b=29/c=94，p<1e-7，**反转**） | **11.679** |
| GPTQ | 0.4871 | — |
- 论述：v1 的缺陷不是"全面校准失败"，而是**组粒度 × 任务类型**的联合边界：g64 长程算术是杀伤区，g128 反而占优。这既限制了结论外推，也解释了为何"更细的分组不一定更好"。

### 6.7 边界条件二：任务轴与协议轴（系统发现）
- Fig 4（核心图）：accuracy–PPL 散点（六方法），显示无单调关系；标注"PPL 最优 → GSM8K 最差"箭头。
- 协议轴现象：板端 few-shot 系统性劣化（1.5B 5-shot 0.4469 < zs 0.5583；3B 5-shot 0.10 复述示例），定性为运行时/模板现象，与量化无关，但直接影响"边缘必须零样本评测"的方法学建议。**[审计 10-25 决策：本条为质性观察——1.5B 5-shot 无落盘 JSON、3B 为 n=50 初步观察，只作叙述不进表格]**
- 3B 系统数据：zs MMLU 0.6037 / GSM8K 0.6700（预转基线，无 NAC 对照——写明为扩展数据而非方法证据）。

### 6.8 系统表征（Table 5，效率三轴）
| prompt 长度 | TTFT (ms) | decode (tok/s) |
|---|---|---|
| 125 | 91.5 | 13.01 |
| 497 | 107.4 | 11.73 |
| 993 | 125.4 | 10.44 |
| 1985 | 181.5 | 7.42 |
- 说明：init ~3.0s；定频四项；**明确声明无 tokens/J**（无电流采样），效率主张仅限上述三轴。

---

## 7. Discussion & Limitations（~900 词）

- **7.1 为什么 PPL 会骗人（机制段）**：PPL 是对数似然的局部平均，对"终止决策"与"多步误差累积"不敏感；GSM8K 的失败发生在低概率尾部（EOS 边际）与误差传播末端，二者在平均似然上被稀释。→ 建议评测清单：任何 4-bit 部署报告须含至少一个长生成任务 + 终止行为检查（如生成长度分布、EOS 率）。
- **7.2 对工具链的可操作建议**：非对称量化要么强制整值 zp，要么在文档中显式声明运行时对 zp 的处理；注入接口（extra_qparams）应校验 zp 域一致性。
- **7.3 修复为何只到一半**：Hessian 补偿改善但未消除长程差距，可能原因——补偿在块内、跨块误差仍累积；校准集（通用语料）不含算术分布；终止行为不受逐层 MSE 目标约束。均为可检验假说，不写成断言。
- **7.4 Limitations（逐条，不粉饰）**
  1. **单平台**（仅 RK3576）；单方法模型（NAC 仅 1.5B）。
  2. **板端修复未验证**：v2 qparams 未做板端注入复评，板端结论止于"诊断 + 对照"，不含"修复后板端指标"（见 §7 P0）。
  3. 校准集敏感性未做消融。
  4. 无能耗（tokens/J）。
  5. v2 成分（int-zp / Hessian）在 sim 轴不可分离。
  6. AWQ 的 PPL 为 16-seq 探针口径，与主表 64-seq 口径不同（表注已声明）。

---

## 8. Conclusion（~250 词）

三段：语义契约破裂（可量化）→ 评测代理失效（可复现，双方法族）→ 部分修复与边界（可操作）。不作 SOTA 主张。

---

## 9. 表图清单与数据源映射

| 编号 | 内容 | 数据源 | 状态 |
|---|---|---|---|
| Table 0 | 协议矩阵 | 本文 §5 | 待制 |
| Table 1 | 板端三配置 | acc_{mmlu,gsm8k}_qwen1.5b_g64*_zs.json | 数据齐 |
| Table 2 | sim 五方法主表 | acc_mmlu_sim_g64.full.json、acc_gsm8k_sim_g64.full.json、ppl_g64_64seq{,_nacv2}.json、awq_probe_ppl16.json、*meta.json | 数据齐 |
| Table 3 | 配对 McNemar 矩阵 | 各 records + 重算 | 部分已算（AWQ/v2 已算），其余需补算统一表 |
| Table 4 | g128 边界 | acc_mmlu_sim_g128.full.json | 数据齐 |
| Table 5 | 系统表征 | perf_qwen1.5b_w4a16_full.json | 数据齐 |
| Fig 1 | 三轴框架图 | 本文 §3.2 | 待绘 |
| Fig 2 | zp 分布 + 等效误差 | zp 分析（需重算并落盘） | **需补算** |
| Fig 3 | κ/α 选择分布 | meta.json（ratio_mean 0.9543） | 数据部分齐 |
| Fig 4 | accuracy–PPL 散点（核心图） | Table 2 | 待绘 |
| Fig 5 | GSM8K 失败模式样例（NAC 数值错乱 / AWQ 终止失效） | records.*.json | 数据齐 |
| Fig 6 | g64 vs g128 | Table 4 | 待绘 |

---

## 10. 投稿前必补实验（按优先级，阻断投稿者标 P0）

- **P0（阻断）板端 int-zp 注入复评**：用 `nacv2_qwen1.5b_g64.qparams.pt` 走注入链路构建 `.rkllm`（构建 ~13min + Optimizing ~5min + export ~45min，qemu 环境），板端跑 MMLU 969 + GSM8K 200。目的：把板端叙事从"诊断 + 对照"升级为"诊断 + 修复"，并首次分离问题 B 的修复效果。
  - **状态：已启动（2026-10-22 22:41，systemd 单元 `p0-intzp`）**。链路：[gen_nacv2_extra_qparams.py](../scripts/gen_nacv2_extra_qparams.py)（转换+审计）→ [p0_intzp_pipeline.sh](../scripts/p0_intzp_pipeline.sh)（构建→冒烟→MMLU→GSM8K），日志 `results/p0_intzp_pipeline.log`。
  - **注入保真度已审计**：196/196 层覆盖、scale 形状与权重 `(out, in/64)` 逐层校验通过、zp 整值率 **1.000000**、越界钳制 7/20,471,808 group（3.4e-7，影响不可测），审计落盘 `results/p0_intzp_qparams_fidelity.json`。注入格式与 E3 对照完全相同（extra_qparams Schema A、同一 build 参数），故 **P0 与 E3 构成严格单变量对照**（唯一变化：float-zp → int-zp）。
  - **评测协议严格复刻 E3**：MMLU zs 969 / GSM8K-Short zs 200，均 32 token、无 taskset、无定频改动。
  - **风险预告（须提前想好两种写法）**：若结果落在 0.2–0.3（介于 0.005 与基线 0.555 之间），按"板端部分恢复，长程差距与 sim 轴一致"写，叙事不破；若仍 ≈0.005，则 int-zp 不是充分条件，问题 B 判定需降级为"必要条件之一"——此时须回到 §6.2 重写归因段。
  - **不得**在未跑前于正文任何位置暗示板端已修复。
- **P1（加分，成本低）**：NAC-v1 + 仅整值 zp 的 sim 消融（校准 ~9min + 评测），用于分离 "zp 舍入" 与 "Hessian" 在 sim 轴的可测差异（预期结论：sim 轴 zp 舍入影响小，与"板端才关键"一致）。
- **P1（加分）**：Fig 2 的 zp 统计与等效误差重算并落盘 JSON（当前仅有均值 10.7% 等汇总数字，需可复现脚本）。
- **P2（可选）**：校准集敏感性（wikitext vs C4 子集）；LongBench 4K 长上下文（板上数小时级）。
- **P3（写作）**：参考文献 ~30 篇（GPTQ/AWQ/SmoothQuant/OmniQuant/rknn-llm/边缘 LLM 2024–2026）；代码与 results JSON 开源声明；cover letter 强调"首个 RK3576 系统评测 + 判别实验可复现"。

---

## 11. 写作纪律（自查清单）

- [ ] 每个形容词有数据背书（尤其 "NPU-Aware"、"Hardware-Faithful"、"efficient"）。
- [ ] 任何数字所在表/图都能在文中指回 `results/*.json`。
- [ ] 跨栈比较一律显式标注"不可绝对值对比"，仅讨论秩与配对差。
- [ ] 平手（p>0.05）一律写"统计平手"，不写"略优/略差"。
- [ ] limitations 的 6 条与正文§7.4 逐条对应，不缺项。
- [ ] P0 未完成前，摘要与结论不得出现板端修复成效。

---

## 12. 建议写作顺序（避免返工）

1. **先定 P0 跑不跑**（若跑，先启动构建，等待期写 §2/§3）。
2. §6 表格骨架 + Table 2/4/5 落数（数据已齐，最快见效）。
3. §3（三轴框架）→ §4（方法）→ §5（setup）：这三节决定审稿人对"方法论贡献"的认可。
4. §6.2/§6.4（两条发现）→ §7.1（机制解释）。
5. §1 → §0.2 摘要 → §8 结论（最后写，确保与实测口径一致）。
6. 最后过 §11 自查清单。

---

## 13. 事故恢复路径（2026-10-23 起，按序阻断）

> **状态更新 2026-10-24**：R1 完成、R2 在 tiny 模型通过。根因从"注入不可能生效"修正为"**注入格式用错**"；权威 Schema 已拿到，下一步是 1.5B 重跑 R2/R3。

**R1 拿到权威 qparams Schema**（✅ 完成，结论修正）：`scripts/probe_tiny_gdq.py` 令 toolkit 以 `gdq` 跑 tiny 并自动落盘 `gdq.qparams`，得到权威格式——**扁平 state-dict**：`{name.weight: int8[-8,7], name.weight_qscale: fp32(out, in/64), bias/layernorm: fp32}`，**无独立 zp 键（zp 已烘焙进 int8）**。此前三次注入的 `{name:{scale,zero_point}}`（float-zp 独立键）从未被识别，这才是 `0 extra qparams already found and loaded!` 的真因。
  - 附带：tk 1.2.3 核心为 Cython `.so`（`rkllm_base/quantizer.*.so`），patch `QuantLinear.calc_qparams/set_qparams` 全程 `calc_calls=0 set_calls=0` → 该注入路线废弃。

**R2 注入生效的硬判据**（✅ tiny 已过第 1/2 条；1.5B 待重跑，第 3 条待补）
1. 日志 `25 extra qparams already found and loaded!`（tiny = 14 int8 weight + 11 fp 参数；1.5B 应为 196 weight 对应计数）；
2. 产物 md5 随 qparams 变化：基线 `db6492…` / scale×8 `162769…` / weight+8 `0c7f2c…` 互异；
3. sim 侧 `apply_qparams` 逐层 dequant 权重 vs NAC 期望的**数值一致性抽检**（float32 舍入级）——上板前保真度确认，待做。

**R3 复现厂商基线**（待执行，门槛已塌）：本地构建不再需要天级 Optimizing（28-block 校准），extra_qparams 命中即跳过；1.5B 仅剩 load + Building model + export。判据不变：≈0.55 → 问题在 g64；≈0.005 → 逐项 bisect（dataset / optimization_level / num_npu_core / toolkit 版本）。

**R4 重做板端对照**（待执行，设计需重估）：NAC-v2 产物改写为新格式（`w_int8=clamp(round(w/s)+zp,-8,7)` + `weight_qscale` + 原样 fp 参数）。注意：新格式只承载 int8 权重 + scale，**float-zp 无法注入**（须舍入烘焙），原 "minmax float-zp vs NAC-v2 int-zp" 对照不再成立；问题 B（NPU 重解释 float-zp）在该接口下**结构性不可检验**，对照变量需重新定义。

**R5 文档回填**（待执行，C2 表述待改写）：板端归因大概率从"float-zp 重解释"改写为"注入格式/语义契约"层面；§6.2 归因链与 C2 待 R3/R4 结果后重写。**在此之前，TECHNICAL_REPORT 的 10-16/10-18 板端结论需标注"待复核"。**

---

## 14. 投稿就绪度审计决策记录（2026-10-25）

> 审计报告：[../subprojects/thesis_audit_2026_10_25/outputs/audit_report.md](../subprojects/thesis_audit_2026_10_25/outputs/audit_report.md)（F1-F3 致命 / M1-M6 大修 / m1-m5 小修 / p1-p3 润色；sim 轴数字对账全过）。

**已拍板决策**：
1. **Venue = TMLR**（方法学/评测论文友好，first-claim 时间戳；MLSys/TCAD 备选）。
2. **few-shot 协议轴发现降格为质性观察**（M3）：1.5B 5-shot 0.4469 / 3B 0.10@n=50 只在 §6.7 作质性叙述（3B 标注 n=50 初步观察），不进表格、不做板端重跑，板端机时留给 R3/R4。
3. **g128 PPL 补 64-seq**（M2）：`taskset -c 4-7 python scripts/ppl_coherence.py --num-seqs 64 --group-size 128 --modes fp16,minmax,nac,gptq --qparams results/nac_qwen1.5b_g128.qparams.pt --gptq-qparams results/gptq_qwen1.5b_g128.qparams.pt --out results/ppl_g128_64seq.json`——**排队条件：R3 板端评测结束后启动**（torch CPU 与 NPU 评测抢 A72 核）；完成后 Table 4 PPL 列换 64-seq 口径，Table 2 表注仅保留 AWQ 16-seq 一处。
4. **M1/M6/F3 已完成**：`results/stats_pairs.json`（22 配对统一重算，gold 序列断言顺序一致，Table 3 唯一来源；新增素材：v2 显著超 minmax GSM8K p=0.025、AWQ vs minmax GSM8K 平手 p=0.248）；`results/env_snapshot.json`（软件版本 + 全部 .rkllm 产物 md5，四旧产物 md5 同值 3bd8c0ba 物证落盘）；git 仓库已建立（.gitignore 排除 models/、*.pt、*.rkllm）。

**投稿前仍开放**：F1（R3/R4/R5 链）、F2（R3/R4 重评统一 32-token 协议并重跑基线）、M4（TECHNICAL_REPORT 行内"待复核"标注，随 R5）、M5（相关工作三线补引，写作期）、m1/m2/m5（正文披露 + 代码归档，写作期）。

