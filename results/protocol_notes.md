# 评测协议说明（D4 补充）

## NAC 模拟量化 PPL 协议（T5）

- 数据：wikitext-2-raw-v1 **test** split（校准用 validation，互不重叠）
- **held-out 协议：16 序列 × 1024 token（token 级均匀抽样）**，bs=16
- 原定 32×1024 在 RK3576 上不可行：torch 2.14.1 aarch64 wheel 的 BLAS 核
  实测仅 ~10-15 GFLOPS（bundled OpenBLAS 无优化 aarch64 kernel；fp32 sgemm
  基准 41.8s/[16,1023,1536]×[1536,8960]，bf16 更慢 3×），单 pass 32×1024 需 ~2.1h
- 16×1024 保持 1024 上下文保真度，PPL 无偏，token 数减半 → 单 pass ~1h
- 三方对比：fp（bfloat16 参考）/ minmax（RTN 基线）/ nac（学习 scale/zp）
- g64 轮复用 g128 的 fp PPL（fp 参考与 group-size 无关，--fp-ppl）

> **口径更新（2026-10-18）**：ppl_coherence.py 已参数化（`--num-seqs`，默认 16 = E4 回归口径；
> 论文口径 64 seq 已排入 nac_v2_watcher 阶段 5，`--modes fp16,minmax,nac,gptq` 四方对比，
> 产物 results/ppl_g64_64seq.json）。上表 16-seq 为 E4 回归校验口径，与 64-seq 论文口径互不覆盖。

## rkllm-runtime 1.2.3 RK3576 build 的 GET_LOGITS 模式实测

- `RKLLMInferParam.mode = RKLLM_INFER_GET_LOGITS (2)` 调用 `rkllm_run` 返回 0，
  但**回调中的 `RKLLMResult.logits` 恒为空**（`logits=NULL, vocab_size=0, num_tokens=0`），
  且行为退化为普通 GENERATE（会产出 token）。GENERATE 模式下回调同样不携带 logits。
- 结论：该 runtime build 未实现经回调的 logits 导出，板端无法做真正的
  last-token logit 比较（需要 x86 toolkit 或修改 runtime 才能拿到分布）。
- 复现方法：以 `RKLLMInferParam(mode=2)` 调 `rkllm_run`，在回调中检查
  `RKLLMResult.logits`（实测 mode=0/2 均为空，mode=2 仍产出 token）。

## 实测结果（969 题，2026-10-06）

| 协议 | MMLU acc | none_pred | ms/题 |
|---|---|---|---|
| 生成式（32 tok + 正则抽取） | 0.4469 | 22.4% 错题中 | ~975 |
| 首 token 字母匹配 | 0.4066 | 23.8% | 832 |

结论：首 token 协议低 4.0 点（无多步纠错机会），两协议均远低于 FP16 参考
0.609 → 量化损伤仍是主因（与 attrib_mmlu.json 一致）。

## 采用的替代协议：first-token letter match

- greedy 生成 1 个 token；若首 token id ∈ {A,B,C,D} 的 token 级变体
  （含前导空格，如 id 32/362 → A），预测该字母，否则 pred=None。
- 与生成式协议（32 token + 正则抽取）对比可量化"协议损耗"上限。
- 脚本：`scripts/eval_mmlu_firsttok.py`，结果 `results/acc_mmlu_firsttok_*.json`。

## 频段 token id（Qwen2.5 tokenizer, vocab 151665）

| token | id | token | id |
|-------|----|-------|----|
| A     | 32 | " A"  | 362 |
| B     | 33 | " B"  | 425 |
| C     | 34 | " C"  | 356 |
| D     | 35 | " D"  | 422 |
