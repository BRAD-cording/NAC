# NAC-RK3576: NPU-Aware Post-Training Quantization for Edge LLM Inference

论文项目：*NPU-Aware Post-Training Quantization for Efficient LLM Inference on Rockchip RK3576 Edge SoC*
目标期刊：Springer Applied Intelligence（备选 IEEE Access）。
统一技术报告（算法/方法论/实验矩阵/进展日志唯一入口）：[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)。
现行发表评审基线：[docs/review_publication_20261017.md](docs/review_publication_20261017.md)（取代 10-15 版）。

## 项目结构

```
nac_paper/
├── configs/            # 模型 / 校准超参
├── src/
│   ├── nac/            # 贡献1：NPU 感知 PTQ 校准（NAC）
│   │   ├── quantize.py       # W4A16 per-group 量化/反量化（Rockchip 公式，uint4 域 [0,15]）
│   │   ├── calibrate.py      # NAC v1：per-layer MSE 优化 scale/zero-point
│   │   ├── hessian.py        # NAC v2：GPTQ 列序 Cholesky 补偿 + 激活加权 κ 网格 + int-zp（问题 A 修复）
│   │   ├── awq.py            # AWQ 基线：激活感知 scale 搜索 + 折叠 + clip 网格（Lin et al. 2023）
│   │   ├── validate_sim.py   # sim 量化验证（apply_qparams / eval_ppl / restore）
│   │   └── export_rkllm.py   # 经 rkllm_toolkit + extra_qparams 导出 .rkllm
│   ├── runtime/
│   │   └── llm_client.py     # ctypes 绑定 librkllmrt.so（板上推理客户端）
│   ├── routing/        # 【已弃用】原贡献2：异构 prefill 路由（P0-3，10-17 决策：砍，零数据不进贡献列表）
│   ├── eval/           # 系统评测
│   │   ├── perf.py           # TTFT / decode tok/s / 端到端延迟
│   │   ├── power.py          # sysfs 功耗采样 + tokens/J
│   │   ├── accuracy.py       # MMLU/GSM8K 任务协议与数据加载（板端 zero-shot；sim GSM8K 用 2-shot CoT）
│   │   └── wikitext_ppl.py   # x86 侧 toolkit get_logits PPL（校准期快速指标）
│   └── utils/
│       ├── data.py           # Wikitext-2 校准集构造 → quant_dataset.json
│       └── freq.py           # CPU/NPU/DDR 定频包装
├── scripts/            # 校准 / 评测 / 流水线编排
├── results/            # 所有实验 JSON 落盘（论文图表数据源）
└── paper/              # 论文骨架 outline.md
```

## 校准与 sim 评测脚本（板上 CPU，taskset -c 4-7）

| 脚本 | 作用 |
|---|---|
| `scripts/calibrate.py` | NAC v1 校准（MSE grid，float-zp） |
| `scripts/nac_v2_calibrate.py` | NAC v2 校准（Hessian 补偿 + 激活加权 + int-zp，问题 A 修复） |
| `scripts/gptq_calibrate.py` | GPTQ 忠实基线（同校准数据 / 同 qparams 格式 / 同块级顺序传播） |
| `scripts/awq_calibrate.py` | AWQ 基线（激活感知 scale 搜索 + 折叠；产物 `.qparams.pt` + `.fold.pt`） |
| `scripts/eval_mmlu_sim.py` / `eval_gsm8k_sim.py` | 板上协议 sim 评测（`--configs label=path`；AWQ 需加 `--fold`） |
| `scripts/ppl_coherence.py` | sim PPL（`--num-seqs` 参数化：E4 口径 16 / 论文口径 64） |
| `scripts/awq_smoke.py` | AWQ 折叠数学冒烟（14 项，不加载大模型） |
| `scripts/nac_v2_watcher.sh` | 自动流水线（systemd --user `nac-v2-wd`）：等队列 → v2 校准 → sim 评测 → PPL 64-seq → AWQ 校准 → AWQ sim 评测（阶段 1-8，产物幂等） |

## 运行环境分工

| 机器 | 角色 | 说明 |
|---|---|---|
| NanoPi RK3576（本机） | 推理 + 评测 + 1.5B/3B CPU 校准 | aarch64, 16GB, librkllmrt.so |
| x86 笔记本（可选） | 7B 校准 + toolkit PPL | rkllm_toolkit 为 x86_64 wheel（cp38/cp310） |

## 快速开始

```bash
# 0) RK3576 端依赖
bash scripts/setup_env.sh

# 1) 定频（保证测量可复现）
sudo bash ../scripts/fix_freq_rk3576.sh   # 或 src/utils/freq.py

# 2) 构造校准数据（Wikitext-2 val, 128 条）
python -m src.utils.data --num-samples 128 --out data/quant_dataset.json

# 3) NAC 校准（v2，本机 CPU，1.5B 约 1.5-2h）
taskset -c 4-7 .venv/bin/python scripts/nac_v2_calibrate.py \
    --model models/Qwen2.5-1.5B-Instruct-hf --group-size 64 \
    --out results/nacv2_qwen1.5b_g64

# 4) 导出 RKLLM（x86 或 toolkit 可用环境；qparams 与步骤 3 产物对齐）
python -m src.nac.export_rkllm --model Qwen/Qwen2.5-1.5B --nac results/nacv2_qwen1.5b_g64.qparams \
    --dtype w4a16 --target rk3576 --out models/qwen2.5_1.5b_nacv2_g64.rkllm

# 5) 板上基线性能
python -m src.eval.perf --model models/qwen2.5_1.5b_nacv2_g64.rkllm --prompt-len 128 512 1024 2048

# 6) 板上精度
python -m src.eval.accuracy --model models/qwen2.5_1.5b_nacv2_g64.rkllm --task mmlu --num-samples 1000

# 7) sim 精度评测（板上 CPU 协议对齐版；多配置 label=path，AWQ 加 --fold）
taskset -c 4-7 .venv/bin/python scripts/eval_mmlu_sim.py \
    --model models/Qwen2.5-1.5B-Instruct-hf \
    --qparams results/nac_qwen1.5b_g64.qparams.pt --group-size 64 \
    --configs minmax,nac --out results/acc_mmlu_sim_g64.json

# 8) 汇总论文表格
python scripts/collect_results.py --results-dir results --out results/tables.md
```

## 排期追踪

原 30 天冲刺排期已终结：现行进展与未决事项以 [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) 的「未决事项」节为准。
原排期中的异构路由（D12–D13）与路由消融（D14 部分）已随 P0-3 砍除而失效；消融聚焦校准/分组两因素。

## 实验矩阵

| 模型 | 量化 | 基线 | 指标 |
|---|---|---|---|
| Qwen2.5-1.5B（3B 附带表征） | W4A16 g64/g128 | FP16(CPU) / GPTQ / AWQ / rkllm默认量化 / NAC(v1/v2) | MMLU、GSM8K、sim PPL、TTFT、decode tok/s、峰值内存 |
