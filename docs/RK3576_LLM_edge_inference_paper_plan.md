# NanoPi RK3576 边缘端 LLM 量化推理论文终极落地方案
## （单设备 · IEEE/Springer Q2 · 30 天录用冲刺）

> **【已归档 2026-10-18】** 本方案为立项期计划，其排期与贡献假设已被实际进展取代（异构路由贡献已砍、双修+AWQ 基线落地）。现行唯一进展入口：`nac_paper/docs/TECHNICAL_REPORT.md`；评审基线：`nac_paper/docs/review_publication_20261017.md`。本文件仅作历史背景保留。

> 硬件：NanoPi RK3576（16GB LPDDR5 + 256GB NVMe，6 TOPS NPU，W4A16）
> 约束：仅边缘推理，无 GPU 训练，仅 CPU 端 PTQ 校准
> 目标：30 天内投稿并获得 Q2 期刊录用通知
> 首选期刊：**Springer Applied Intelligence**（JCR Q2, IF 3.5, 初审 14–45 天, AI 方向, 录用友好）
> 备选期刊：IEEE Access（Q1, 初审 ~28 天, 录用率约 20%）

---

## 〇、时间可行性硬分析

要在 30 天内拿到录用通知，必须满足：

| 环节 | 占用天数 | 说明 |
|---|---|---|
| 实验 + 撰写 | 1–18 天 | 18 天内必须完成全部实验和论文 |
| 投稿 | 第 18 天 | 越早投越好 |
| 初审（first decision） | 19–32 天（约 14 天） | Applied Intelligence 最快 14 天 |
| 小修 + 返回 | 33–35 天 | 2 天内完成小修 |
| 录用通知 | 第 35–40 天 | 若小修则编辑 3–5 天内给最终决定 |

**关键结论**：必须第 18 天前投稿，且论文质量要让审稿人只提小修（minor revision）。若初审超过 20 天，录用会延到第 40 天以后。若被 Applied Intelligence 拒稿，立即转投 IEEE Access（Q1，虽录用率低但审稿同样约 4 周）。

---

## 一、论文核心（只做一个强贡献，保证一次过审）

### 题目
*NPU-Aware Post-Training Quantization for Efficient LLM Inference on Rockchip RK3576 Edge SoC*

### 核心故事（一句话）
现有 PTQ 方法（GPTQ/AWQ）在 GPU 上校准，部署到 Rockchip NPU 的 W4A16 流水线后精度掉得比预期多；我们提出 NPU 感知校准，用 NPU 真实量化路径做损失，在 3 个模型上恢复 1–3% 精度，同时系统评测 RK3576 上 LLM 推理的性能/功耗/精度权衡。

### 为什么这个选题稳
1. **贡献清晰**：一个方法 + 一组系统评测，审稿人容易判断价值
2. **实现简单**：校准脚本 ~300 行 Python，1 周可完成
3. **数据好出**：RK3576 是热门平台，但缺乏系统性量化评测，容易填补空白
4. **scope 匹配**：Applied Intelligence 收 AI 应用+系统优化；IEEE Access 全领域收

---

## 二、贡献与方法

### 贡献 1：NPU 感知 PTQ 校准（NPU-Aware Calibration, NAC）

**问题**：GPTQ/AWQ 在 GPU 上用 FP16 模拟 INT4 量化误差，但 Rockchip NPU 执行 W4A16 时：
- 使用 per-group 量化（group size = 64 或 128），每组独立 scale + zero-point
- 反量化公式与 GPU 模拟存在系统性偏差
- 结果：校准期显示精度损失 <1%，实际 NPU 部署后损失 3–5%

**方法**（CPU 端完成，约 300 行 PyTorch）：
```python
# 伪代码：对每一层 l
fp16_out = layer_l(fp16_input)                          # CPU FP16 前向，记录每层输出
for g in [64, 128]:                                      # 对比 group size
    # 用 NPU 真实 W4A16 反量化公式量化权重
    q_weight, scale, zp = quantize_w4a16_group(layer.weight, group=g)
    quant_out = layer_l(fp16_input, weight=dequantize(q_weight, scale, zp))
    # 优化 scale/zero-point 使量化输出逼近 FP16 输出
    scale, zp = optimize(loss=MSE(quant_out, fp16_out), scale, zp)
# 保存校准后的 scale/zp，转换为 RKNN 格式
```

**NPU 真实反量化公式**（Rockchip W4A16）：
```
W_dequant[i] = scale[group(i)] * (W_int4[i] - zero_point[group(i)])
其中 group(i) = i // group_size
```

**预期**：MMLU 精度恢复 1–3%，3B/7B 模型收益大于 1.5B。

### 贡献 2（轻量，凑完整性）：CPU/NPU 异构 Prefill 路由

**观察**：实测发现长序列 prefill（>512 token）时 CPU（llama.cpp）比 NPU 快（内存带宽瓶颈）。

**方法**：实测 RK3576 上 prefill 延迟随序列长度变化，找到阈值 L*，L > L* 时 prefill 走 CPU，decode 始终走 NPU。

**预期**：长序列端到端延迟降 15–25%。此部分作为附加优化，非核心，但能增加论文完整性。

---

## 三、实验设计（18 天可跑完）

### 3.1 设备与模型

| 设备 | 模型 | 量化配置 | 说明 |
|---|---|---|---|
| RK3576 | Qwen2.5-1.5B | W4A16 g64/g128 | 主平台 |
| RK3576 | Qwen2.5-3B | W4A16 g64/g128 | 主平台 |
| RK3576 | Qwen2.5-7B | W4A16 g128 | 16GB 可放下（权重≈4.5GB） |

> 为什么选 Qwen2.5：Rockchip 官方 rknn-llm 支持最好，部署最稳定。

### 3.2 数据集

| 用途 | 数据集 | 规模 | 跑一次耗时 |
|---|---|---|---|
| 校准 | Wikitext-2 val | 128 条 | <10 分钟（CPU） |
| 精度 | MMLU（5-shot，子采样 1k 题） | 1k | 3B 模型约 4–6 小时（NPU） |
| 推理 | GSM8K（8-shot） | 200 条子采样 | 约 2 小时 |
| 长上下文 | LongBench 4K 子集 | 100 条 | 约 3 小时 |

> MMLU 子采样 1k 题：57 个科目每科约 18 题，统计上足够。过夜跑。

### 3.3 评测指标

- **精度**：MMLU、GSM8K 准确率
- **性能**：TTFT、decode tokens/s、端到端延迟（prompt 长度 128/512/1024/2048）
- **资源**：峰值内存、功耗（RK3576 sysfs powercap）、能效 tokens/J
- **量化对比**：FP16 vs GPTQ vs AWQ vs NAC（本文方法）

### 3.4 基线（5 组）

| 基线 | 说明 |
|---|---|
| FP16 (CPU) | 精度上界（仅 1.5B/3B 能在 16GB 跑 FP16） |
| GPTQ W4A16 | GPU 校准，转 RKNN 部署到 NPU |
| AWQ W4A16 | GPU 校准，转 RKNN 部署到 NPU |
| 纯 NPU 默认量化 | rknn-llm 自带量化（无校准） |
| **NAC（本文）** | NPU 感知校准 + 异构路由 |

### 3.5 消融（3 组）

1. 去掉 NAC 校准（用 GPTQ 校准 + 本文路由）→ 验证校准贡献
2. 去掉异构路由（全程 NPU）→ 验证路由贡献
3. group size g64 vs g128 → 验证分组选择

---

## 四、30 天逐日排期

### Week 1（Day 1–7）：环境 + Baseline

| 天 | 任务 | 交付物 |
|---|---|---|
| D1 | RK3576 刷机、装 rknn-llm、llama.cpp | 环境可用 |
| D2 | 跑通 Qwen2.5-1.5B/3B/7B W4A16 推理 | 模型可加载、能生成 |
| D3 | 部署 GPTQ/AWQ 量化模型到 NPU | 量化转换流程跑通 |
| D4 | 跑 FP16 精度（1.5B/3B，CPU）+ GPTQ/AWQ 精度（NPU） | baseline 精度数据 |
| D5 | 测 prefill/decode 延迟（128/512/1024/2048） | baseline 性能数据 |
| D6 | 测功耗、内存 | baseline 资源数据 |
| D7 | 整理 baseline 数据，确认 NPU 精度 gap 存在 | 动机数据 |

**Week 1 里程碑**：确认 GPTQ/AWQ 部署到 NPU 后精度比校准期低 ≥2%（若 gap 不明显，论文动机弱，需调整选题）。

### Week 2（Day 8–14）：NAC 实现 + 精度验证

| 天 | 任务 | 交付物 |
|---|---|---|
| D8–9 | 实现 NPU 感知校准脚本（CPU 端 PyTorch） | 可运行的校准代码 |
| D10 | 跑 NAC 校准 1.5B/3B/7B，转 RKNN | 校准后模型 |
| D11 | 跑 NAC 模型 MMLU/GSM8K 精度 | 精度提升数据（目标 +1–3%） |
| D12 | 实现 CPU/NPU 异构路由 | 调度器代码 |
| D13 | 测路由后延迟（长序列） | 延迟下降数据（目标 -15%） |
| D14 | 跑消融实验 | 消融数据 |

**Week 2 里程碑**：NAC 精度提升 ≥1%，路由长序列延迟下降 ≥15%。任一不达标，聚焦达标的那个贡献写。

### Week 3（Day 15–18）：论文撰写 + 投稿

| 天 | 任务 | 交付物 |
|---|---|---|
| D15 | 写 Introduction + Related Work | 论文前半部分 |
| D16 | 写 Methodology + Experiments | 论文主体 |
| D17 | 写 Results + Conclusion，画图 | 完整初稿 |
| D18 | 润色、查重、投稿 Applied Intelligence | **投稿完成** |

**Week 3 里程碑**：第 18 天 24:00 前完成投稿。

### Week 4（Day 19–30）：审稿应对

| 天 | 任务 |
|---|---|
| D19–28 | 等待初审（Applied Intelligence 中位 14 天，可能 D32 左右出结果） |
| 收到审稿意见 | 若小修：2 天内修回；若大修：尽量满足，若拒稿转 IEEE Access |
| D29–30 | 若已收到小修意见，完成修改并返回 |

**若 D30 前未收到审稿意见**：论文仍在审，录用通知会在 D32–40 到达。这是 IEEE/Springer 的客观周期，无法再压缩。

---

## 五、论文写作要点（确保小修而非大修）

1. **Introduction 写清楚动机**：用 D4–7 的 baseline 数据量化"GPU 校准 vs NPU 实测"的精度 gap，让审稿人一眼看到问题
2. **Method 简洁**：NAC 核心公式一段话说清，伪代码 20 行内
3. **实验全面**：3 模型 × 5 基线 × 4 指标，表格完整，不留明显漏洞
4. **讨论局限性**：主动说明"仅评测 RK3576 单平台"，审稿人反而觉得诚实
5. **References 充分**：引用 GPTQ、AWQ、SmoothQuant、rknn-llm 相关工作，约 30 篇

> Applied Intelligence 要求论文 8000–12000 词，约 15–20 页双栏。D15–18 四天写完时间紧，建议 D12–13 就开始写 Introduction。

---

## 六、风险与应急预案

| 风险 | 概率 | 预案 |
|---|---|---|
| GPTQ/AWQ 转 RKNN 后精度 gap <2%，动机弱 | 中 | 扩大对比到更多模型/数据集，或改测"长上下文精度退化"作为动机 |
| NAC 精度提升 <1% | 中 | 转强调"同精度下 g128 比 g64 更快"，或只做系统评测论文（无方法贡献，纯 benchmark） |
| 异构路由收益不显著 | 低 | 长序列（1024/2048）收益必然明显，重点测长序列 |
| rknn-llm 对 7B 支持有 bug | 中 | 只跑 1.5B/3B，7B 作为"尝试"写入局限性 |
| Applied Intelligence 初审 >20 天 | 中 | 等待，录用延到 D40 左右，仍属快速发表 |
| 被 Applied Intelligence 拒稿 | 低–中 | D19 立即转投 IEEE Access（Q1，审稿 ~4 周） |
| 18 天写不完论文 | 低 | D12 开始写，优先写实验部分（数据已有） |

---

## 七、不使用 GPU 的可行性

- **校准**：CPU 端 PTQ，仅优化 per-group scale/zero-point，1.5B 模型全部层约 30 分钟
- **FP16 精度**：1.5B/3B 可在 RK3576 的 16GB CPU 内存跑 FP16（不需要 GPU）
- **NPU 推理**：rknn-llm 直接在 RK3576 NPU 上跑量化模型
- **评测**：MMLU 1k 题 3B 模型约 4–6 小时，过夜跑
- **完全不需要 GPU**，一台 RK3576 + 可选的 x86 CPU 笔记本（加速校准）即可

---

## 八、一句话执行方针

**第 18 天前把"3 模型 × 5 基线 × NAC 校准 + 异构路由"的论文投到 Applied Intelligence，剩下 12 天等小修。**

---

*文档日期：2026-09-28（终极版：单 RK3576 · IEEE/Springer Q2 · 30 天冲刺）*
