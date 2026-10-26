# R3→R4 Bisect：本地构建 vs 厂商预转差异分析（2026-10-27）

> 背景：R3 判别分确认 ≈0.005 分支——本地 toolkit 1.2.3 plain 构建（`models/local_w4a16_plain_x86.rkllm`）板端 GSM8K zs 0.015 / MMLU zs 0.4974，厂商预转（`models/Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm`）0.5550 / 0.5583。同一算法族（normal/w4a16/rk3576）下板端表现天差地别，且崩坏模式与 10-23"注入构建"同型。本文档记录离线 bisect 证据、假设清单与 R4 最小对照实验设计。

## 1. 离线证据

### 1.1 字节级 diff 地图（/tmp/rkllm_diffmap.py，8MB 分块）

两文件大小完全相同（1,410,933,676 B）。

| 区域偏移 | 状态 | 判读 |
|---|---|---|
| 0 – 448 MiB | **字节级全同** | 头部元数据（eos/bos/max_context）+ fp16 embedding 均一致 |
| 448 – 464 MiB | mid/low（10–50% 差异） | layer 0 附近部分差异 |
| 464 – 496 MiB | **全同（32MB）** | 疑为某层/某张量完全一致 |
| 496 – 1120 MiB | high（~100% 差异，576MB+） | int4 权重 payload 大面积不同（非仅 scale 表差异） |
| 1120 – 1128 MiB | mid | 过渡带 |
| 1128 MiB – 1345.6 MiB | **全同（217.6MB）** | 尾部（final norm / lm_head 区）一致 |

**判读**：(a) 模型源、tokenizer、dtype、所有元数据完全一致（前 448MB 全同）；(b) 差异全部集中在量化层权重区，且是 payload 级差异而非零星 scale 差异 → 差异产生于**校准/量化/打包阶段**；(c) 部分区域全同（32MB 带、层 0 部分）→ 不是整体换了一种布局，更像局域的量化决策不同。

### 1.2 厂商 model card（HF: HanzoHuang/Qwen2.5-1.5B-Instruct-RKLLM）

- **Conversion toolkit: RKLLM Toolkit 1.2.3**（与我们相同 → 版本差异假设对 1.5B 降级）
- Max context length: 4096（与我们相同）
- RK3576 w4a16；另有 w4a16_g128 选项（说明其 w4a16 默认组粒度，g128 需显式指定）
- 注意：3B 基线（VRxiaojie）是 1.1.4 转的，版本假设对 3B 仍开放，但 1.5B 是主战场

### 1.3 本地构建配置（~/r3x86/scripts/r3_plain_build.py）

`load_huggingface(hf, None, "cpu", "float32", None, True)` → `build(do_quantization=True, quantized_dtype="w4a16", quantized_algorithm="normal", optimization_level=1, target_platform="rk3576", num_npu_core=2, dataset="data/quant_dataset.json")` → export（eos 151645 / bos 151643 / add_bos False / max_context 4096）。

- dataset = rknn-llm 官方示例自带 `data/quant_dataset.json`：128 条 WikiText 风格英文样本（291KB）
- optimization_level=1 与官方示例 export 脚本一致（examples/rkllm_multimodel_demo/export/export_rkllm.py 亦用 opt=1）
- 校准数据集与 1.1.4 版 qparams 生成用的是同一份

### 1.4 toolkit save_qparams 产物（results/tk_w4a16_full.qparams，626KB）

torch zip 格式，141 条 = 28×(q/k/v_proj.bias + input_layernorm + post_attention_layernorm) + model.norm，全部为 fp32 张量副本。**不含任何 MLP/o_proj/embed 权重的 scale/zp**——save_qparams 只回存"透传张量"，不是完整量化记录；无法由此反查 toolkit 自算 scale/zp，也无法直接驱动 sim 伪量化栈。

## 2. 假设清单（按嫌疑排序）

| # | 假设 | 证据 | 嫌疑 |
|---|---|---|---|
| B1 | **校准数据集不同**（vendor 用自备/更大的多域数据集；我们用 128 条英文 wikitext）。toolkit 1.2.3 的 normal 并非纯权重 minmax——Optimizing 707s/层 × 28 层 = 5.5h，说明跑了大量校准前向（激活 clamp 搜索类），数据分布直接进入量化决策 | 权重区 payload 级差异 + 层 0（最先接触 embedding 激活）部分差异 | **高** |
| B2 | **quantized_algorithm 不同**（vendor 或用 ld_learning 等激活感知算法） | 同上，会改变全层量化决策；但厂商 model card 未声明算法 | 中 |
| B3 | **optimization_level 不同**（vendor 或用 3；op 融合改变张量打包布局 → 即使数值相同也会呈现 ~100% 字节差异；但 layer0 部分差异 + 32MB 全同带更像量化决策差异而非纯布局重排） | 官方示例用 1；无法确认 vendor 用几 | 中 |
| B4 | **toolkit 版本不同** | 1.5B 厂商 model card 明写 1.2.3 → 对 1.5B 降级（3B 仍开放） | 低（1.5B） |
| B5 | 元数据/模板/eos/bos/max_context 不同 | 前 448MB（含头部）字节级全同 → **排除** | 已排除 |
| B6 | 模型源/tokenizer/加载 dtype 不同 | embedding fp16 区全同 → **排除** | 已排除 |

## 3. R4 最小对照实验设计

每轮成本：笔记本构建 ~6h（Optimizing 5.5h + export）+ scp 回传 ~1h + 板端 GSM8K zs 200 评测 ~1.5h（32-token 协议）≈ 8–9h/轮。**优先做 x86 侧可判别的实验**（md5/diff 地图即可判别，不必等板端分数）。

- **R4-a（首选，1 轮）**：唯一改动 = 换校准数据集（用更大/多域数据，如 wikitext-2 全量采样 512 条 + 中英混合；或 vendor 社区常用的大数据集）。判别：(a) 与 vendor md5 全同 → 直接锁定 vendor 数据集（中彩）；(b) diff 地图模式改变（权重区从 high 变 low/部分差异）→ 数据集进入量化决策被证实，板端评测这版看 GSM8K 是否恢复；(c) 与 R3 本地版 diff 仍同模式 → 数据集无关，转 B2/B3。
  - **10-27 执行修正**：多域数据使构建 OOM（13GB VM 上限；128 条多域仍爆，R3 128 条 wikitext 能过 → 多域内容推高校准内存峰值，与条数弱相关；`vm.swappiness=100` 后稳定）。最终校准集 = **64 条多域**（gsm16/code8/zh16/wt24，`data/quant_dataset_r4a64.json` md5 9077d495…；512/128 条留档）。B1 检验效力不变：与 R3 的唯一主差异仍是数据分布；条数缩减对 normal 校准统计影响小。
- **R4-b（次选）**：唯一改动 = optimization_level=3。判别：diff 地图是否整体重排（支持 B3）。
- **R4-c（可选）**：唯一改动 = quantized_algorithm（若 1.2.3 提供 normal 之外的算法）。
- **辅助**：构建时把 toolkit 的量化中间日志（每层 clamp/scale 统计）打开（若 API 可选 verbose），与 diff 地图逐层对齐可把差异定位到具体张量级，避免整轮板端盲试。

## 4. 对论文叙事的影响（outline 修订预留）

- 10-23 板端崩塌的最终解释路径改为：**本地 toolkit 1.2.3 构建路径在某配置下产生 NPU 端不可用的量化产物**（与注入/zp 无关）；E3"minmax float-zp 对照同崩"重释为"两次本地构建同崩"。
- 问题 B（NPU zp 重解释）降级为次要假设；H2 重写为"构建配置敏感性问题"。
- 板端 Table 1 重建判据：任何本地构建必须先通过"厂商基线复现"（GSM8K zs ≥ 0.5）才有资格做校准对照注入——这是 R4 之后的 R5。
- 若 B1 证实：新增一条系统级发现——**校准数据分布对 RK3576 NPU 产物可用性的一票否决效应**（128 条英文 wikitext → GSM8K 0.015；数据集工程 = 边缘部署链的一等公民），这本身可写进 §6.7/§7。
