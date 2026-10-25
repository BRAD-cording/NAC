# NAC-RK3576 统一技术报告

> ⛔ **2026-10-23 待复核警告**：P0 复评发现**板端注入链路从未生效**。三次注入构建产物 md5 完全相同（`3bd8c0ba…`），tk 1.2.3 日志均为 `0 extra qparams already found and loaded!`。故本文档 **10-16 起的全部板端结论（§3.2、E3 对照、问题 B "zp 重解释"判定、板端 0.5614/0.005）暂标"待复核"**，不得引用至论文。sim 轴（§3.1）数据不受影响。恢复路径见 [paper/outline.md §13](../paper/outline.md)。复核完成后本节删除并出 v1.5。

**版本**：2026-10-22（v1.4：AWQ 链收官——探针裁定 EOS 退化任务特异、MMLU 0.4396 与 gptq/v2 平手、GSM8K 0.000；v2 PPL 11.962 落定，全实验矩阵关账；v1.3=v2 MMLU 收官 0.4469 追平 gptq、PPL 64-seq 扩样关账、AWQ 校准 bug 修复重启；v1.2=10-18 E3/E4 收官回填、问题 B 坐实、双问题判定闭环；v1.1=E2 部分结果回填；v1=2026-10-16 zp 根因调查期快照）
**范围**：NAC 算法、双轨评测方法论、实验矩阵总览、进展日志。本文档是论文 §3–§5 的写作底稿，也是新协作者/评审者的唯一入口。
**评审基线**：[review_publication_20261017.md](review_publication_20261017.md)（现行，含双问题叠加判定）；历史链：[spec_zp_rootcause.md](spec_zp_rootcause.md)（10-15 版 Q2+ 评审 review_q2_20261015.md 已于 10-18 删除，信息并入本报告与 10-17 版评审）。

---

## 1. NAC 算法（贡献 1）

### 1.1 量化核（Rockchip NPU W4A16 公式）

RK3576 NPU 的 W4A16 反量化路径（[quantize.py](../src/nac/quantize.py)）：

```
W_dequant[i] = scale[g] * (W_int4[i] - zp[g]),  g = i // group_size
```

- 非对称 uint4 量化域 `[0, 15]`，per-group 独立 `scale / zp`
- 量化：`q = clamp(round(W/scale + zp), 0, 15)`；反量化：`W' = scale*(q - zp)`
- **zp 为 float（软连续语义）**——这是本项目的核心语义声明，也是当前根因调查焦点（见 §4）

### 1.2 NAC 主算法（grid 模式，[calibrate.py](../src/nac/calibrate.py)）

GPTQ 式顺序量化传播 + AWQ 式激活加权 per-group 网格搜索，板上 CPU 可行：

1. **捕获**：FP 路径前向，pre-hook 记录每 block 输入 hidden（顺序传播用）与每 Linear 的按通道激活幅值均值 `|x|`（加权用）
2. **逐 block 量化**（输出传播到下一 block）：
   - 对每个 Linear 做 per-group 网格搜索截断比例 `κ ∈ {1.0, 0.95, ..., 0.4}`（组间独立取最优），最小化激活加权误差：

     ```
     Σ_g min_κ  a ⊙ (W - dequant(q(W; scale_κ, zp_κ); scale_κ, zp_κ))²
     ```

   - dequant 权重写回活体模型，块前向一次得到下一 block 的真实输入
3. **导出**：`.qparams.pt`（每 Linear 的 per-group scale/zp）+ `.meta.json`

设计取舍：
- **为什么网格搜索而非 Adam-STE**：Adam-STE 逐块可学习版本（`--mode adam`）需 x86+GPU，板上实测 82 TFLOP/块 × 28 块 ≈ 87h 不可行；grid 模式纯权重张量运算，无需 block forward
- **为什么 float32**：A72/A53 无 bf16 硬件支持
- **与 GPTQ 的关系**：GPTQ 用 Hessian 近似补偿，NAC 用激活加权截断搜索；两者均按 NPU 反量化公式生成 qparams，语义对齐

### 1.2b NAC-v2（问题 A 修复，[hessian.py](../src/nac/hessian.py)，10-17）

E2 三方分化（§3.2）暴露 v1 目标函数缺陷：float-zp 语义族（minmax/NAC）在 sim GSM8K 同崩而 GPTQ 独活——激活加权 MSE + 截断网格缺长程误差补偿。v2 在 v1 框架上引入：

1. **GPTQ 列序 Cholesky 误差补偿**（块内 layer-parallel Hessian，同 GPTQ 基线路径，保证基线可比性）
2. **激活加权 κ 网格**（保留 v1 的截断搜索，与补偿项联合）
3. **integer-zp 默认开**（对齐问题 B 处置：NPU 重解释嫌疑下整值 zp 语义安全，且与 GPTQ 语义一致）

校准脚本 `scripts/nac_v2_calibrate.py`；冒烟通过（int-zp 格点往返严格 0 误差，float-zp 往返 9.7e-3——佐证 int-zp 默认）；排入 `nac-v2-wd` 自动链（等 sim 队列出内存窗口后校准 + sim GSM8K/MMLU 评测）。

### 1.3 zp 语义边界声明（关键）

| 语义域 | zp 处理 | 谁在用 |
|---|---|---|
| **sim 伪量化**（校准/评测/PPL） | float 连续值直接参与 `scale*(q-zp)` | `validate_sim.py` 伪量化写回 |
| **NPU 硬件**（推测，待证） | 可能对 zp **取整或重解释**（H1，见 §4） | 板上 librkllmrt |

NAC 产出的 zp 99.84% 非整数（|frac| 均值 0.25，范围 [-0.063, 15.105]）；GPTQ 产出 100% 整数 zp。**若 NPU 取整 zp，NAC 的等效权重误差 ≈ 权重幅度的 10.7%（均值，最大约 3.9×）**——H1 的定量基础。

> **10-17 更新**：E2 三方分化后根因升级为**双问题叠加**——问题 A（上：校准目标缺误差补偿，sim 忠实语义下即可见）+ 问题 B（本节：NPU zp 重解释嫌疑，E3 板端对照裁决）。v2 与 AWQ 均已默认整值 zp。详见 [review_publication_20261017.md](review_publication_20261017.md) §3。

---

## 2. 双轨评测方法论

### 2.1 量化器轴区分（P0-2 的解法）

所有表格与叙述必须显式区分三条独立轴：

1. **量化器来源轴**：toolkit 内置量化（minmax/GPTQ）vs NAC 外部注入（extra_qparams）
2. **执行栈轴**：sim 伪量化（x86 float32）vs 板端 NPU 真机（librkllmrt）
3. **评测协议轴**：任务（MMLU/GSM8K/PPL）× shot（zs/few-shot）× 采样数（969/200/16）

任何跨轴比较都是无效比较——历史教训：曾把 sim 量化器与板端量化器混入同一列。

### 2.2 板端协议（[accuracy.py](../src/eval/accuracy.py)）

- **必须 zero-shot**。few-shot 存在系统性退化：toolkit 1.5B GSM8K zs 0.555 → 2-shot 0.23（崩）；3B 5-shot MMLU 0.10。已定性为板端现象（推测与 tokenizer/对话模板处理有关），与 NAC 无关
- MMLU：969 题（1000 采样池去重）生成式，`--fewshot-num 0`
- GSM8K：200 题 zs 生成式
- 注入模型需 runtime ≥1.2.3（识别 `W4A16_G64`）

### 2.3 sim 协议（[validate_sim.py](../src/nac/validate_sim.py) / [eval_mmlu_sim.py](../scripts/eval_mmlu_sim.py)）

- 伪量化写回：`apply_qparams(model, gs, mode, ...)` 逐层替换权重（**float zp 连续语义**）
- MMLU 969 题 zs generative（32 tok）+ firsttok 双指标；PPL 用 wikitext-2 test heldout（16 seq × 1024）
- **配对陷阱**：records 的 `i` 字段是批计数不是题号；跨配置配对必须按**数组位置**

### 2.4 统计方法

- 配对显著性：McNemar 检验（χ² 连续性修正：`χ² = (|n10-n01|-1)²/(n10+n01)`，`p = erfc(sqrt(χ²/2))`，statsmodels 不可用时手动实现）
- 区间：bootstrap CI95
- 平手判据：p > 0.05 且差异 < CI 宽度

---

## 3. 实验矩阵总览（g64, Qwen2.5-1.5B）

### 3.1 精度矩阵

| 配置 | MMLU sim zs (969) | MMLU 板端 zs (969) | GSM8K sim zs (200, E2) | GSM8K 板端 zs (200) | PPL sim (16seq) |
|---|---|---|---|---|---|
| fp16（sim 伪量化=无量化） | 0.4634（终值） | — | — | — | 10.627 |
| toolkit minmax（sim） | **0.4675** | — | 0.075（16/200） | — | 11.738 |
| toolkit GPTQ（sim） | 0.4520 | — | **0.305**（200/200 终值；早段快照偏高：16→0.5625、64→0.333） | — | — |
| NAC（sim） | 0.3932（-7.43pp vs minmax，p<1e-4） | — | 0.070（16/200） | — | **11.408**（优于 minmax） |
| toolkit 基线 w4a16（板端） | — | 0.5583 | — | 0.555 | — |
| NAC 注入（板端） | — | **0.5614**（McNemar p=0.878，平手） | — | **0.005**（灾难） | — |
| E3 minmax float-zp 对照注入（板端） | — | 0.5614（平手） | — | **0.005**（与 NAC 同崩） | — |
| **g128: toolkit minmax（sim）** | 0.3622 | — | — | — | 11.988 |
| **g128: NAC（sim）** | **0.4293**（+6.71pp vs minmax，McNemar b=29/c=94，p<1e-7，**反转**） | — | — | — | **11.679**（优于 minmax） |
| **g128: toolkit GPTQ（sim）** | **0.4871** | — | — | — | — |
| **NAC-v2（sim，int-zp+Hessian）** | **0.4469**（969/969 终值；显著超 v1 nac 0.3932，b=98/c=46 p=2.1e-5；与 gptq 0.4520 平手 −0.51pp p=0.716；与 minmax/fp16 无显著差 p=0.106/0.13） | — | **0.150**（200/200；显著超 v1 nac 0.070，McNemar b=26/c=10 p=0.012；仍显著低于 gptq 0.305，b=15/c=46 p=1.2e-4） | — | **11.962**（64-seq，stage 9 落定 10-22；较 v1 11.893 +0.6%，仍优于 gptq/minmax） |
| **AWQ（sim，激活加权 scale+fold）** | 0.4396（acc_ftok 0.4365；与 gptq −1.24pp p=0.369 **平手**、与 v2 −0.72pp p=0.579 平手；显著低于 minmax −2.79pp p=0.033 / fp16 −2.37pp p=0.038） | — | **0.000**（0/64 主动截断，none_rate=0；**EOS 终止退化**，探针裁定真实校准行为，见 §3.2） | — | **11.226**（16-seq 探针口径，三方法最优；fold_only 10.627 精确复现 fp16） |

> **g128 边界条件（10-20 落定）**：v1 NAC 对 minmax 的 MMLU 劣化在 g64 为 −7.43pp，g128 反转为 **+6.71pp**（p<1e-7），与 g128 PPL 更优（11.679 vs 11.988）方向一致——问题 A 的杀伤集中在 **g64 粒度 + 长程算术（GSM8K）**场景，而非全面的校准失败。论文叙事：v1 目标函数的缺陷有明确边界（组粒度 × 任务类型），v2 的 Hessian 补偿针对的正是该边界场景。

> **PPL 64-seq 扩样（10-21 落定，P2-7 论文口径关账）**：fp16 11.075 / minmax 12.225 / nac(v1) 11.893 / gptq 12.168（64 seq × 1024，[ppl_g64_64seq.json](../results/ppl_g64_64seq.json)）。与 16-seq 口径方向一致：v1 NAC PPL 仍最优、gptq 反最差——"任务精度-困惑度"双轴权衡成立：GPTQ 的 Hessian 补偿偏向任务关键误差方向而非整体失真最小（GSM8K 独活与 PPL 最差并存），v1 NAC 的 PPL 占优 + 任务崩溃共同指向"校准目标缺长程补偿"而非整体失真。v2 PPL 已由 stage 9 补齐（10-22 落定 [ppl_g64_64seq_nacv2.json](../results/ppl_g64_64seq_nacv2.json)）：**11.962**，较 v1 11.893 +0.069——Hessian 补偿以近零 PPL 代价实现（v2 行入上表）。

### 3.2 核心矛盾与假说

**矛盾三联**（截至 10-16）：
1. sim：NAC MMLU 显著劣化（-7.43pp）但 PPL 更优
2. 板端：NAC MMLU 平手（≈toolkit）但 GSM8K 灾难（0.555 → 0.005）
3. GSM8K 失败模式：预测全为小数（'3.0'/'1.0'/'50.0'），长度中位数 3，无空预测——模型仍能生成，但算术能力被摧毁

**H1（主假说，E3 已直接坐实）**：NPU 对 float zp 取整/重解释。注入等效权重误差 ~10.7%（GSM8K 精确算术敏感 → 崩；MMLU 四选一容错 → 平手）。支持证据：GPTQ 全整数 zp 板端正常；sim 连续语义下 NAC PPL 更优（sim 与 NPU 语义分歧的直接表现）；**E3 minmax float-zp 对照板端同崩（10-18，见下）**。

**H2（已排除）**：toolkit 未真正应用 extra_qparams。反证：若 toolkit 用自家量化，GSM8K 不会崩（自家 normal 量化等价于基线行为）。

**10-17 E2 部分结果 → 双问题叠加**：sim GSM8K 三配置分化——gptq **0.305**（200/200 终值，10-18 落定）vs minmax **0.075**、nac **0.070**（同检查点）。sim 协议被证明有效（gptq 显著优于 minmax/nac 约 4×，任务信号可传递）→ minmax/NAC 的 sim 崩溃是真实的校准质量问题，即**问题 A（校准目标缺长程误差补偿）**；板端 zp 重解释嫌疑降为**问题 B**（E3 板端对照裁决）。原判别矩阵"NAC 崩 + minmax 正常 → H3"之外出现第三组合：float-zp 语义族同崩、GPTQ 整数 zp 独活。修复须双管齐下（NAC-v2 已实施，§1.2b）。
> 口径注记：sim gptq 0.305 仍低于板端 toolkit w4a16 的 0.555——差异来源候选：8-seq 校准 vs toolkit 全量校准、sim 批处理/padding、fp32 与 NPU 数值路径；此差距本身是"sim-板端桥接"（P0-2 轴区分）需讨论的对象，不影响 E2 的判别逻辑（判别靠三配置相对分化，不靠绝对值）。

**10-18 E3 收官 → 问题 B 坐实（双问题判定闭环）**：minmax float-zp 对照注入板端 GSM8K zs = **0.005**（与 NAC 注入 0.005 同级灾难，失败模式一致）、MMLU zs = **0.5614**（与基线 0.5583/NAC 0.5614 平手）。对照与 NAC 崩溃谱系完全一致 → 板端崩溃**非 NAC 特有**，而是 float-zp 注入在 NPU 上普遍失真，**问题 B（NPU 重解释 float zp）由 E3 直接坐实**（对照也崩分支，判别矩阵落定）。E2 坐实问题 A + E3 坐实问题 B，双问题叠加从预判（10-17）变为双证据闭环。修复对应关系：问题 B → integer-zp（NAC-v2 与 AWQ 均已默认整值 zp）；问题 A → 长程误差补偿（NAC-v2 Hessian 项），v2 sim GSM8K 为最终验收。

**10-20 v2 首判（问题 A 修复部分有效，验收未全额达标）**：v2 sim GSM8K 终值 **0.150**（200/200，38297s）——McNemar 配对显著超 v1 nac 0.070（b=26/c=10，**p=0.012**），Hessian 长程补偿方向确认有效；但仍显著低于 gptq 0.305（b=15/c=46，**p=1.2e-4**），"追近 gptq"未达成。结合 g128 边界条件（上文），v1/v2 与 GPTQ 的差距集中于 g64 长程算术场景；若 AWQ 对照同样不及 gptq，论文叙事按 review §三 预案收敛为"**系统评测 + 语义发现 + 部分修复 + 边界刻画**"（g128 反转是边界刻画的核心证据）。MMLU v2 与 AWQ 链进行中。

**10-21 v2 MMLU 收官（问题 A 修复在 MMLU 全额追平 gptq）**：v2 sim MMLU 终值 **0.4469**（969/969，41106s）——McNemar 显著超 v1 nac 0.3932（**+5.37pp，b=98/c=46，p=2.1e-5**）；与 gptq 0.4520 差 −0.51pp **统计平手（b=58/c=63，p=0.716）**，对 minmax（−2.06pp，p=0.106）与 fp16（−1.65pp，p=0.13）亦无显著差。至此修复边界刻画完成：v1 在 MMLU 上的 −7.43pp 劣化被 v2 Hessian 补偿**全额恢复至 gptq 同级**，v1/v2 与 GPTQ 的残余差距仅存于 **GSM8K 长程算术**（0.150 vs 0.305，p=1.2e-4）——"短上下文任务全恢复、长程算术部分恢复"。"部分修复"叙事证据链完整；AWQ 对照链因校准脚本 bug 顺延一日（见 §5 10-21/10-22 行）。

**10-22 AWQ 收官（EOS 退化任务特异；"任务精度-PPL 脱钩"升级为方法族现象）**：AWQ sim GSM8K acc=**0.000**（0/64 主动截断，none_rate=0，[acc_gsm8k_sim_g64_awq.json](../results/acc_gsm8k_sim_g64_awq.json)）——records 抽查显示生成连贯、CoT 步骤正确（gold=83 题算出 "The answer is 83"），但**不终止、跨题续写**致答案抽取从续写文本错位取数。甄别探针（PPL 分解，16-seq E4 口径，[awq_probe_ppl16.json](../results/awq_probe_ppl16.json)）：fold_only **10.627**（精确复现 fp16 10.627 → 跨块折叠端到端无损，折叠数学正确）、awq_full **11.226**（优于同口径 nac(v1) 11.408 / minmax 11.738，三方法最优）→ **无实现 bug，EOS 终止退化为真实校准行为**（判定锚三分支的中间分支）。AWQ sim MMLU **0.4396**（969/969，[acc_mmlu_sim_g64_awq.full.json](../results/acc_mmlu_sim_g64_awq.full.json)）：与 gptq −1.24pp（b=81/c=69，p=0.369）、与 v2 −0.72pp（b=62/c=55，p=0.579）**双平手**，显著低于 minmax（−2.79pp，p=0.033）/ fp16（−2.37pp，p=0.038）→ 崩溃为 **GSM8K 长生成（EOS 终止）任务特异**，首 token/分类任务无损。与 E2"v1 NAC PPL 占优 + 任务崩溃"并案：**激活加权目标族（NAC-v1 截断网格 / AWQ scale 搜索）在 PPL/首 token 轴无损甚至占优、在长程生成轴崩溃**——PPL-下游脱钩从单例（v1 NAC）升级为方法族现象，为问题 A"校准目标缺长程补偿"的归因提供第二族独立证据；同时反衬 v2/GPTQ 的 Hessian 补偿正是长程生成存活的关键分量。v2 PPL 64-seq 11.962 同日落定（stage 9，+0.6% vs v1）——**全实验矩阵（g64 sim 五方法 × MMLU/GSM8K/PPL + g128 三配置 + 板端注入双评）关账**。

### 3.3 甄别实验（E2/E3/E4 全部收官，2026-10-18）

| 编号 | 实验 | 状态 | 判别意义 |
|---|---|---|---|
| E2 | sim GSM8K × {minmax,gptq,nac} | **收官（10-18）**：gptq **0.305**（200/200）/ minmax 0.075 / nac 0.070 | gptq 显著独活 → **问题 A 成立**（校准质量）。sim gptq 0.305 vs 板端 0.555 的绝对值差距另行桥接讨论（P0-2） |
| E3 | minmax float-zp 对照注入构建 + 板端 MMLU/GSM8K zs | **收官（10-18）**：板端 GSM8K **0.005**（与 NAC 同崩）/ MMLU 0.5614（平手） | 对照与 NAC 同崩 → **问题 B 坐实**：float-zp 注入在 NPU 普遍失真（修法=integer-zp） |
| E4 | PPL 一致性（当前代码×当前 qparams 复算） | **收官（10-18）**：minmax 11.738 / nac 11.408，与 10-08 首测完全一致 | 排除 qparams.pt 过期/代码变动的混淆 |

E2×E3 组合落定判别矩阵：**float-zp 语义族 sim/板端双崩 + GPTQ 整数 zp 双正常 → 双问题叠加（A 校准目标 + B 板端 zp 重解释）**。组合矩阵详见 [spec_zp_rootcause.md](spec_zp_rootcause.md)。

---

## 4. 注入链路（路径 C，已打通）

### 4.1 extra_qparams 格式（已破解）

```python
torch.save({f"model.layers.{b}.{mod}": {"scale": fp32(out, n_groups),
                                        "zero_point": fp32(out, n_groups)}},
           path)   # HF 命名；196 层全量；缺省层走 toolkit 默认量化
```

### 4.2 构建流程（[mmctrl_pipeline.sh](../scripts/mmctrl_pipeline.sh) 同模板）

```python
llm = rkllm.RKLLM()
llm.load_huggingface(model_dir, ...)            # fp32 原始模型（带权重）
llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
          quantized_algorithm="normal", optimization_level=1,
          target_platform="rk3576", num_npu_core=2,
          dataset="data/quant_dataset.json",
          extra_qparams="<qparams 文件>")         # 用户 qparams 注入
llm.export_rkllm("models/....rkllm")
```

- 耗时：Building ~13min、Optimizing ~5min、export 流式 ~45min（板上 qemu 环境）
- 产物 `models/nac_qwen1.5b_g64_nac.rkllm`（1.54GB），NPU runtime 正确加载
- **qparams 大文件禁止落 /tmp**（重启即失，10-16 教训）：统一存 `results/*.qparams`

### 4.3 已证伪路线（留档防复走）

- **calc_qparams 猴子补丁**：normal 量化主路径从不调用它，文件注入是唯一通道（[build_nac_inject.py](../scripts/build_nac_inject.py) 留档）
- **gdq 算法**：强制 float32+CUDA，板上 CPU 不可行（[build_gdq_probe.py](../scripts/build_gdq_probe.py)）
- **--optim 0**：仅调优化阶段内部优先级，不能跳过 Optimizing

### 4.4 构建病理（重要约束）

`Optimizing 0/28` 卡死（内存无界增长 → swap 雪崩）**绑定「dequant 重写模型」输入**：原始模型 × 用户 qparams 全阶段稳定通过；任何 dequant 重写模型（含普通 w4a16 配置）在当前 qemu 环境必然卡死。构建输入必须是 fp32 原始模型。

---

## 5. 进展日志（决策链）

| 日期 | 事件 |
|---|---|
| 10-08 | NAC g64 校准产物 `results/nac_qwen1.5b_g64.qparams.pt`（12:45）；sim PPL 三组（fp16 10.627 / minmax 11.738 / nac 11.408） |
| 10-13 | GPTQ 基线 qparams（13:31） |
| 10-14 | 机器三次重启（09:48/12:19/13:50），用户单元丢失，实验停摆 25.5h |
| 10-15 | 队列断点恢复（nac 512/969 续跑）；Q2+ 再评审定稿（P0-1/2/3）；路径 C 侦察：extra_qparams 格式破解、补丁路线证伪、gdq 不可行；**注入构建成功**（196 层全量，全阶段通过）；板上 NAC MMLU zs = 0.5614（平手，P0-1 闭合） |
| 10-16 | 板端 NAC GSM8K zs = 0.005（灾难）发现；few-shot 退化定性（板端必须 zs）；sim 969 终值复核（NAC MMLU 劣化 + PPL 优势的矛盾确认）；**zp 分布分析坐实 H1 前提**（NAC zp 99.84% 非整数 vs GPTQ 100% 整数）；minmax 对照甄别编排（E2/E3/E4 自动化）；12:42 再次重启 → 13:10 全量恢复；fp16 段收尾中 |
| 10-17 | 发表评审定稿（review_publication_20261017：双问题叠加判定、收录概率表、题目分析）；**P0-3 路由已砍**（零数据不进贡献列表，outline/README 同步）；**AWQ 基线已实施**（src/nac/awq.py + scripts/awq_calibrate.py，激活感知 scale 搜索 + GQA 轨道折叠 + clip 网格 + 整值 zp；fold.pt 只存 scale 向量 ~1.5MB，eval `--fold` 同一 fold_block_ 重建保证位级一致；冒烟 14/14 通过）；**PPL 扩样已实施**（ppl_coherence.py `--num-seqs` 参数化，64-seq 论文口径入 watcher 阶段 5）；**双修启动**（问题 A = NAC-v2 hessian.py 冒烟通过 + nac-v2-wd 编排排入阶段 2-4；问题 B = E3 自动裁决）；E2 首批数据落盘（gptq 0.5625 / minmax 0.075 / nac 0.070 @16/200）；能效叙事收敛延迟/吞吐/内存三轴 |
| 10-18 | 文档收敛：冗余清理（30 天计划、10-15 版评审删除+引用更新、tables/protocol_notes 过时标注）；TECHNICAL_REPORT 升 v1.1（NAC-v2 方法论小节 §1.2b、双问题叠加入 §3.2）；E2 gptq 段收官 acc=0.305（200/200，38573s）——早段快照 0.5625 系小样本偏高，终值仍 4× 于 minmax/nac，问题 A 判定不变；**E4 PPL 复核收官**（minmax 11.738 / nac 11.408，与 10-08 首测完全一致，排除 qparams 过期混淆）；**E3 收官：minmax float-zp 对照板端 GSM8K 0.005 同崩、MMLU 0.5614 平手 → 问题 B 坐实，双问题叠加判定闭环**；watcher 恢复队列时 systemd-run 缺 `--working-directory` 致 nac-queue 127 静默失败（08:19-22:42 停摆 ~14h），修复 mmctrl_watcher.sh 后以正确 cwd 重启队列（22:42），stage 0/1 records 续跑零损失，GPTQ g128 校准 22:51 开跑；报告升 v1.2 |
| 10-19 | GPTQ g128 校准收官（00:41，~1.8h）；MMLU g128 三配置开跑（实测 42.3s/题，单配置 ~11.4h，远超注释乐观估计）；minmax 终值 0.3622、nac 终值 0.4293 落盘 |
| 10-20 | **MMLU g128 三配置收官**（gptq 0.4871；配对：nac vs minmax **+6.71pp 反转**，McNemar b=29/c=94，p<1e-7——v1 劣化仅在 g64 出现，g128 边界条件入 §3.1）→ **queue ALL DONE 11:11**；v2 g64 校准自动开跑并收官（11:11-12:34，int-zp）；**v2 sim GSM8K 终值 0.150**（23:16，200/200，38297s）——显著超 v1 nac 0.070（p=0.012）仍显著低于 gptq 0.305（p=1.2e-4），**问题 A 修复部分有效、验收未全额达标**（判定入 §3.2）；MMLU v2 自动开跑（23:18，~11.4h） |
| 10-21 | v2 sim MMLU 收官 **0.4469**（10:54，969/969，41106s）：显著超 v1 nac（+5.37pp，p=2.1e-5）、**与 gptq 统计平手**（p=0.716）——问题 A 修复在 MMLU 全额追平（判定入 §3.2）；PPL 64-seq 四模式落定（fp16 11.075 / minmax 12.225 / nac 11.893 / gptq 12.168，P2-7 关账，入 §3.1 注）；20:10 **awq_calibrate.py 首跑崩溃**：激活 hook 以模块全名（self_attn.q_proj）计数而 x_bar 按后缀名查询 → KeyError，watcher 退出（unit failed） |
| 10-22 | **AWQ 冒烟分层验证揪出 3 处 bug，修复后重启**：(1) awq_calibrate 激活 hook 以模块全名计数而 x_bar 按后缀查询 → KeyError（10-21 崩溃主因）；(2) fold_block_ 的 σ 行除未摊平 [n_kv,hd] 轨道张量 → 形状崩溃；(3) **生产者 bias 未同步÷scale**（Qwen2.5-1.5B qkv 带 bias：v 行÷σ 后 A·b 项无法被 o 列乘补偿）——纯静默，2 块冒烟通过仍会毒化全部 AWQ 数据，靠 block 级折叠等价性断言揪出；修复后逐 scale 家族分解验证 + 真实 scale 全过（rel ~6.7e-6，ones 对照精确 0）；qparams 键名对齐生态（self_attn.*/mlp.* 全路径，产物 shape/int-zp/有限性实测 OK）。实测校准 ~47s/块（28 块 ~25min，远快于注释 2.5h）；watcher 幂等重启（reset-failed + systemd-run 正确 cwd），AWQ 链续跑。**晚间收官**：AWQ GSM8K 05:07 中止于 64/200（acc=**0.000**、生成连贯但不终止——EOS 退化嫌疑）；05:32 PPL 分解探针落定（fold_only 10.627 精确复现 fp16、awq_full 11.226 三方法最优）→ **无实现 bug，EOS 退化为真实校准行为**，截断结果按裁决保留；恢复 watcher（幂等跳过 stage 1-7）→ AWQ MMLU 18:52 收官 **0.4396**（vs gptq p=0.369 / vs v2 p=0.579 双平手）→ 崩溃定性 **GSM8K 长生成任务特异**（判定入 §3.2）；stage 9 v2 PPL 21:16 落定 **11.962**（+0.6% vs v1）。**全实验矩阵关账，叙事按 review §三 预案收敛**（报告升 v1.4） |

**未决事项**：
1. ~~E2 收官 → E4 PPL 复核 → E3 对照 → 问题 B 落定~~ **全部收官（10-18）**：问题 B 已坐实（E3 对照同崩），判定闭环
2. grid-fix 重校准产物（10-08 18:27 日志）与 qparams.pt（12:45）的时间序待查——当前统一报告按 12:45 版本 qparams 口径
3. ~~P0-3 路由 / AWQ 缺席 / PPL 扩样~~ 已全部处置（10-17，见日志行）——结果回收亦全部完成：~~MMLU g128 三配置~~（10-20 入 §3.1）；~~v2 sim GSM8K/MMLU、PPL 64-seq~~（10-21 入 §3.1/§3.2）；~~AWQ 链 + v2 PPL~~（10-22 入 §3.1/§3.2）
4. 双修验收：问题 B 已由 E3 坐实且修法已内建（NAC-v2/AWQ 默认整值 zp）；问题 A v2 已终判（10-20/21）：GSM8K 0.150 部分有效（超 v1 p=0.012、未及 gptq p=1.2e-4）+ MMLU 0.4469 追平 gptq（p=0.716）——残余差距精确界定于长程算术单轴；**AWQ 对照已落定（10-22）：MMLU 0.4396 与 gptq/v2 双平手、GSM8K 0.000（EOS 退化，任务特异）——AWQ 同族未解长程生成，叙事定判收敛为"系统评测 + 语义发现 + 部分修复 + 边界刻画"**（g128 反转 + AWQ 方法族脱钩为边界刻画双证据）
5. sim 队列：~~GPTQ g128 校准 → MMLU g128 三配置~~ **queue ALL DONE（10-20 11:11）**；v2 链：~~校准~~ ~~sim GSM8K 0.150~~ ~~sim MMLU 0.4469~~ ~~PPL 64-seq~~ 收官（10-21）；AWQ 链：~~校准~~ ~~sim GSM8K 0.000（探针裁定 EOS 退化，0/64 截断）~~ ~~sim MMLU 0.4396~~ ~~v2 PPL 11.962~~ **全链收官（10-22 21:16），全部自动实验完毕**——余下仅论文写作（outline 重写，review §六 步骤 6）
6. 投稿前打磨（承接自已删除的 10-15 版评审 §三 P1-6/P2-7）：(a) 长上下文证据——LongBench 4K 从未运行，3B 5-shot 长 prompt 崩坏现象可升格为系统评测第二数据点（板上数小时级）；(b) 校准集敏感性消融（wikitext-2 vs C4 子集，可选加分）；(c) 参考文献清单（~30 篇 2024–2026 边缘 LLM/NPU 部署）+ 代码开源声明
