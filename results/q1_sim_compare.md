# sim GSM8K：截断协议下的对比

| config | n | acc (last-num) | acc (answer-is) | truncated | median chars |
|---|---|---|---|---|---|
| fp16 | 200 | 0.5300 | 0.5400 | 0.86 | 896 |
| gptq | 16 | 0.6875 | 0.7500 | 0.25 | 2477 |

**配对 McNemar（相对 fp16）**

| comparison | n | fp16 | other | Δ (pp) | b | c | p |
|---|---|---|---|---|---|---|---|
| fp16 vs gptq | 16 | 0.7500 | 0.6875 | -6.25 | 2 | 1 | 1 |

旧口径（不截断、末数字）参考：GPTQ 0.305 / min-max 0.075 / NAC-v1 0.070 / NAC-v2 0.150。
若新口径下 fp16 与各量化方法差距 <10pp，则 Finding 1 需改写为「协议放大」而非「校准质量」。
