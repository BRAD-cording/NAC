# 论文表格汇总（自动生成）

> **过时标注（2026-10-18）**：本表生成于 2026-10-06，其后产出的 969 题 sim MMLU、sim GSM8K 三配置（minmax 0.075 / nac 0.070 / gptq 进行中）、板端 zp 根因甄别系列、NAC-v2/AWQ 结果均未纳入；Table 4 对应的异构路由贡献已砍（P0-3，10-17）。现行结果汇总以 `docs/TECHNICAL_REPORT.md` 为准，待主实验链（E3/v2/AWQ）落地后统一重生成。

## Table 1: Accuracy (MMLU / GSM8K, RKLLM W4A16 on NPU)

| 配置 | MMLU (生成式) | MMLU (首token) | GSM8K |
|---|---|---|---|
| qwen1.5b_w4a16 | 0.45 | 0.41 | 0.23 |

## Table 2: Performance (median, NPU)

| 配置 | TTFT@128 (ms) | decode@128 (tok/s) | TTFT@2048 (ms) | decode@2048 (tok/s) | Power (W) | Energy/token (J) |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct_RK3576_w4a16 | 99.9 | 11.29 | -- | -- | -- | -- |
| qwen1.5b_w4a16 | 115.7 | 8.66 | 205.3 | 5.54 | -- | -- |

## Table 3: NAC sim-quant PPL (wikitext-2 test, CPU fp32)

| 配置 | fp16 | minmax | NAC | Δminmax | ΔNAC | NAC改善% |
|---|---|---|---|---|---|---|

## Table 4: Prefill routing threshold (CPU vs NPU)【已弃用：P0-3 路由贡献 10-17 砍除】

| 模型 | L* | 结论 |
|---|---|---|

## LaTeX（Table 1 对应）

```latex
\begin{table}[t]
\centering\caption{Accuracy of W4A16 quantized models on RK3576 NPU.}
\begin{tabular}{lccc}
\toprule
Config & MMLU & MMLU (first-tok) & GSM8K \\
\midrule
qwen1.5b-w4a16 & 0.45 & 0.41 & 0.23 \\
\bottomrule
\end{tabular}
\end{table}
```
