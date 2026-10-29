# 投稿合规清单（Q1 目标）

面向 **Elsevier 系（FGCS / JSA / Neural Networks）** 与 **TMLR**。每项给出当前状态与文件位置；`[ ]` 表示尚未完成。

## A. 通用硬性项

| 项 | 要求 | 状态 | 位置 |
|---|---|---|---|
| 正文 | 英文、结构完整、图表编号连续 | ✅ 初稿 17 页 | `paper/draft_v0.1.md` / `paper/build/draft_v0.1.pdf` |
| 摘要 | ≤ 250 词，无引用、无缩写 | ✅ 231 词 | draft `## Abstract` |
| 关键词 | 5–6 个 | ✅ 5 个 | draft `**Keywords**` |
| Highlights | 3–5 条，每条 ≤ 85 字符 | ❌ 待写 | 见下方 §D |
| 图表 | 单独文件、≥300 dpi、矢量优先 | 🟡 SVG 矢量已有，PDF 内嵌 | `paper/figs/*.svg` |
| 参考文献 | 统一风格（Elsevier numbered） | 🟡 15 条，格式接近但需按刊校准 | draft `## References` |
| 数据可用性声明 | 必填 | ❌ 待写 | 见 §E |
| 利益冲突声明 | 必填 | ❌ 待写 | 见 §E |
| 作者贡献（CRediT） | Elsevier 必填 | ❌ 待写 | 见 §E |
| 覆盖信 | 1 页，说明贡献与不符 SOTA 主张 | ❌ 待写 | 见 §F |
| 推荐审稿人 | 3–5 名（可选） | ❌ 待定 | — |

## B. 与证据相关的强制项（审稿人会查）

| 项 | 状态 | 说明 |
|---|---|---|
| 每个数字可追溯到归档文件 | ✅ | `analysis/check_claims.py`，33/33 PASS |
| 生成全文归档 | ✅ 新跑；🟡 历史 sim 记录截断 300 字符 | `results/*.records*.json` |
| 协议字段落盘（`max_new_tokens`/抽取规则/截断策略） | ✅ | summary JSON |
| 工件 md5 与 dtype | ✅ | `reproduce/MANIFEST.md` |
| 复现命令 | ✅ | `reproduce/README.md` |
| 代码可获取声明 | 🟡 需选仓库/匿名镜像 | — |
| 已知缺口显式声明 | ✅ | README §6、draft §8 |

## C. 各刊差异

| 刊 | 特殊要求 | 我们需补 |
|---|---|---|
| **FGCS**（Elsevier, Q1） | Highlights + 图形摘要可选；强调系统贡献 | Highlights；第二模型族 |
| **JSA**（Elsevier, Q1） | 系统架构导向；建议 8–12k 词 | 正文偏短（约 5.6k 词），需扩方法与讨论 |
| **Neural Networks**（Q1） | 方法/理论导向 | 需机制实验（误差传播/EOS logit） |
| **TMLR** | 无页数限制；强调可复现与诚实 | 基本就绪，补 sim 重跑 |

## D. Highlights 草案（每条 ≤ 85 字符）

```
- Calibration objectives rank oppositely by perplexity and by long-form reasoning (84)
- A 32-token cap zeroes out even the vendor model: 0.0000 vs 0.5550 at 512 tokens (84)
- Local artifacts score 0.535-0.565 at 512 tokens, tied with the vendor (p >= 0.69) (85)
- Last-number extraction inflates a 1.3x gap into an apparent 4.1x gap (75)
- Rebuilding with the documented default loses 6.1 MMLU points to the vendor (79)
```

（字符数按含空格计，投稿前用工具复核。）

## E. 声明草案

**Data availability.** All evaluation records, the claims ledger and the build scripts are archived with the paper; every reported number is re-derived by `analysis/check_claims.py`. The vendor model artifacts are redistributed under their original licences; our locally built `.rkllm` files are available on request.

**Declaration of competing interest.** The authors declare no competing interests.

**CRediT.** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing — original draft, Writing — review & editing: [to be assigned by the authors].

## F. 覆盖信要点（3 段）

1. **问题**：边缘 NPU 上的 4-bit 部署几乎都在模拟里评估，而"模拟=部署"的假设缺少测量学检验。
2. **发现**：校准目标在 PPL/选择题上不可区分，却在长程生成上相差 4×；其中约 3/4 由"是否在答完后停止"造成，且一个 32-token 上限能让参考模型归零；同厂商默认参数重建会丢 6.1pp MMLU，只有换算法才追平。
3. **边界**：单平台、单模型族；不主张 SOTA；给出可复现协议、检查清单与全部归档。

## G. 投稿前 10 步

1. `analysis/check_claims.py` 全绿；
2. 补齐 sim fp16 锚点后重跑 §6.1 表；
3. 按目标刊模板重排（Elsevier `elsarticle` 或 TMLR 模板）；
4. 图表导出为独立 PDF/EPS；
5. 核对参考文献（DOI、年份、会议名）；
6. 写 Highlights / 声明 / 覆盖信；
7. 英文学术润色（母语或工具）；
8. 查重（含自引）；
9. 补充材料打包（records + 脚本 + MANIFEST）；
10. 提交前再跑一次本清单 A–C。
