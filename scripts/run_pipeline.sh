#!/usr/bin/env bash
# 基线 + NAC + 消融 全套板上评测流水线（D4–D14）
# 用法: bash scripts/run_pipeline.sh <rkllm模型路径> <标签> [task精度样本数]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?用法: run_pipeline.sh <rkllm模型路径> <标签> [num_samples]}"
TAG="${2:?缺少标签}"
NUM="${3:-1000}"
TOK="${TOKENIZER:-models/tokenizer}"
PY="${PYTHON:-.venv/bin/python}"

echo "==> [1/4] 定频 + 记录频率"
sudo "$PY" -m src.utils.freq --mode performance

echo "==> [2/4] 性能评测（128/512/1024/2048, 含功耗）"
"$PY" -m src.eval.perf --model "$MODEL" --tokenizer "$TOK" --power \
    --out "results/perf_${TAG}.json"

echo "==> [3/4] 精度评测（zero-shot，与既有基线协议一致）"
"$PY" -m src.eval.accuracy --model "$MODEL" --task both \
    --fewshot-num 0 --num-samples "$NUM" --out "results/acc_${TAG}.json"

echo "==> [4/4] 路由阈值扫描"
"$PY" -m src.routing.threshold_scan --model "$MODEL" --tokenizer "$TOK" \
    --out "results/route_threshold_${TAG}.json"

echo "==> 完成: $TAG"
