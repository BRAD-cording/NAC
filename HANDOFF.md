# 项目交接说明（2026-10-30 01:10）

本文件是"当前状态快照"，供隔一段时间回来的人（或下一个会话）在 5 分钟内接手。

## 1. 一句话状态

论文从"证据链断裂、正文零字"推进到**可编译的 15 页英文初稿 + 完整证据包 + 两条自动化实验链**；
核心事实已全部核验，剩余工作是等待板端/sim 长跑实验出分后回填两个表，以及补第二模型族。

## 2. 最重要的三个事实（都已实测，勿再引用旧文档）

1. **板端 GSM8K 的"崩塌"是 32-token 截断伪影。** 三个产物 @32 → @512：厂商 0.0000 → 0.5550；本地重建 0.0150 → **0.5350**（p=0.69）；组量化构建（曾被当作"NAC 崩塌"证据）0.0050 → **0.5650**（p=0.88，answer-is 与末数字完全相同）。唯一变量是生成上限。
2. **"NAC / E3 / NAC-v2" 三个板端产物是同一个字节文件**（md5 `3bd8c0ba…`，实测），注入从未生效；且它们的 dtype 是 `W4A16_G64`，与对照组 `W4A16` 不同。
3. **末数字抽取把"不停止"记成"算错"。** 各方法在答完后继续写下一题的比例：GPTQ 11%、NAC-v1 30%、NAC-v2 37%、min-max 38%、AWQ 53%；改用 answer-is 指标后，GPTQ 对 NAC-v1 的优势从 4.1× 降到约 1.3×。

因此：outline 的 C2、Table 1、E3、R3–R4 bisect 全部作废；**仍然成立**的是 sim 轴五方法矩阵、g128 符号翻转、板端 MMLU 构建路径差异（−6.09pp，p=3e-4；gdq 平手 p=0.18）。

## 3. 交付物在哪

| 文件 | 内容 |
|---|---|
| `docs/review_q1_20261029_v2.md` | Q1 就绪度评审（147 条级证据复核、致命问题、修复路线、预判审稿意见 O1–O10） |
| `paper/draft_v0.1.md` | 英文初稿（15 节 + 5 表 + 5 附录，约 5.6k 词） |
| `paper/build/draft_v0.1.pdf` | 编译好的 PDF（15 页，含图 1–4） |
| `paper/build/build_weasy.sh` | 一键重建 PDF（pandoc → HTML → WeasyPrint） |
| `reproduce/MANIFEST.md` | 全部 `.rkllm` 的 md5/dtype/角色（含"三文件同 md5"物证） |
| `reproduce/README.md` | 每个表的复现命令 + 明确缺口 |
| `analysis/check_claims.py` | 29 条断言的可执行证据账本（当前 29/29 PASS） |
| `results/q1_board512_summary.md` | 板端 512 复测汇总（自动生成） |
| `results/q1_sim_summary.md` | sim 重跑汇总（自动生成） |

## 4. 正在跑的任务（无人值守）

| 任务 | 触发/状态 | 产物 |
|---|---|---|
| 板端 512 队列 `q1_board512_queue.sh` | 运行中（`q1bd8` g64 构建） | `acc_gsm8k_q1bd8_zs512.json` 等 |
| 板端汇总 `q1_postprocess.sh` | 等 `results/q1_board512.DONE` | 汇总 md + git 提交 |
| sim 重跑 `q1_sim_queue.sh` | 等板端 DONE（避免抢核） | `acc_gsm8k_sim_g64_stop_*.json`（fp16/GPTQ/NAC-v2） |
| sim 汇总 `q1_sim_postprocess.sh` | 等 `results/q1_sim.DONE` | 汇总 md + git 提交 |
| WSL 构建 R4-e（grq） | Optimizing ~3/28，预计 05:15 | `local_w4a16_r4e.rkllm` |

## 5. 出分后要做的两件事（各 10 分钟）

1. `results/q1_board512_summary.md` 出现后：把 `q1bd8`（g64 构建 @512）的结果补进 `paper/draft_v0.1.md` §6.2.1 的双行对照表，并在 §6.5 说明 g64 产物在 512 下是否健康。
2. `results/q1_sim_summary.md` 出现后：用 fp16 锚点替换 Table 1 中的 `[[TBD: fp16 anchor]]`，并用截断后的 acc 更新 §6.1 的 4.1×/1.3× 表述（若截断后差距变化）。

## 6. 仍未做（按价值排序）

1. **第二模型族**（3B 或 0.5B）——Q1 系统类期刊最可能要求的一项。注意 3B 的 `.rkllm` 已被删除，需重新下载并校验 sha256。
2. **R5：用正确 schema 重建 1.5B NAC 注入产物**——目前板端没有任何真实的 NAC 臂。`scripts/q1_board512_queue.sh` 的注释与 §9 P0-a′ 有步骤说明。
3. 校准集敏感性消融、LongBench 长上下文（成本高，可作 future work）。
4. 投稿前按 `review §7` 的对账表逐行核对，并跑一次 `analysis/check_claims.py`。

## 7. 重建 PDF

```bash
wsl -d Ubuntu -- bash /mnt/d/work/nacpaper/nac_paper/paper/build/build_weasy.sh
```

依赖：pandoc 3.1.3 + WeasyPrint 70（已装在 WSL 用户目录），字体 DejaVu。图在 `paper/figs/*.svg`，由 `analysis/figs.py` 纯 Python 生成（无 matplotlib 依赖）。
