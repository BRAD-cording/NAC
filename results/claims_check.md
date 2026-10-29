| ID | 断言 | 来源 | 实测 | 期望 | 结果 |
|---|---|---|---|---|---|
| B1 | vendor GSM8K @32tok == 0.0000 | `acc_gsm8k_qwen1.5b_w4a16_zs32.json` | 0.0 | 0.0 | PASS |
| B2 | vendor GSM8K @32tok 记录 max_new_tokens == 32 | `acc_gsm8k_qwen1.5b_w4a16_zs32.json` | 32 | 32 | PASS |
| B3 | vendor GSM8K @512tok == 0.5550 | `acc_gsm8k_qwen1.5b_w4a16_zs.json` | 0.555 | 0.555 | PASS |
| B4 | NAC 注入 GSM8K @32tok == 0.0050（伪影，非退化） | `acc_gsm8k_qwen1.5b_g64nac_zs.json` | 0.005 | 0.005 | PASS |
| B5 | E3 minmax 注入 GSM8K @32tok == 0.0050 | `acc_gsm8k_qwen1.5b_g64mmctrl_zs.json` | 0.005 | 0.005 | PASS |
| B6 | 本地构建 GSM8K @512tok == 0.5350（与 @32 的 0.0150 对照） | `acc_gsm8k_qwen1.5b_plain_x86_zs512.json` | 0.535 | 0.535 | PASS |
| B7 | 本地构建 @512 记录 max_new_tokens == 512 | `acc_gsm8k_qwen1.5b_plain_x86_zs512.json` | 512 | 512 | PASS |
| B8 | g64 构建（曾被标为 NAC 崩溃）GSM8K @512tok == 0.5650 | `acc_gsm8k_q1bd8_zs512.json` | 0.565 | 0.565 | PASS |
| B9 | g64 构建 @512 的 answer-is 指标同为 0.5650（无续写污染） | `acc_gsm8k_q1bd8_zs512.json` | 0.565 | 0.565 | PASS |
| B10 | 第三个产物（symv2）@512 == 0.5600 | `acc_gsm8k_q1symv2_zs512.json` | 0.56 | 0.56 | PASS |
| B11 | 第三个产物 answer-is 同为 0.5600 | `acc_gsm8k_q1symv2_zs512.json` | 0.56 | 0.56 | PASS |
| S1 | sim fp16 MMLU 0.4634 | `acc_mmlu_sim_g64.json` | 0.4634 | 0.4634 | PASS |
| S2 | sim minmax MMLU 0.4675 | `acc_mmlu_sim_g64.full.json` | 0.4675 | 0.4675 | PASS |
| S3 | sim GPTQ MMLU 0.4520 | `acc_mmlu_sim_g64.full.json` | 0.452 | 0.452 | PASS |
| S4 | sim NAC-v1 MMLU 0.3932 | `acc_mmlu_sim_g64.full.json` | 0.3932 | 0.3932 | PASS |
| S5 | sim NAC-v2 MMLU 0.4469 | `acc_mmlu_sim_g64_nacv2.full.json` | 0.4469 | 0.4469 | PASS |
| S6 | sim AWQ MMLU 0.4396 | `acc_mmlu_sim_g64_awq.full.json` | 0.4396 | 0.4396 | PASS |
| G1 | sim minmax GSM8K 0.075 (n=200, 512tok) | `acc_gsm8k_sim_g64.full.json` | 0.075 | 0.075 | PASS |
| G2 | sim GPTQ GSM8K 0.305 (n=200) | `acc_gsm8k_sim_g64.full.json` | 0.305 | 0.305 | PASS |
| G3 | sim NAC-v1 GSM8K 0.070 (n=200) | `acc_gsm8k_sim_g64.full.json` | 0.07 | 0.07 | PASS |
| G4 | sim NAC-v2 GSM8K 0.150 (n=200) | `acc_gsm8k_sim_g64_nacv2.full.json` | 0.15 | 0.15 | PASS |
| G5 | sim AWQ GSM8K 0.000 且 n==64 | `acc_gsm8k_sim_g64_awq.json` | 64 | 64 | PASS |
| P1 | g64 PPL fp16 11.0747 | `ppl_g64_64seq.json` | 11.0747 | 11.0747 | PASS |
| P2 | g64 PPL minmax 12.2251 | `ppl_g64_64seq.json` | 12.2251 | 12.2251 | PASS |
| P3 | g64 PPL nac 11.893 | `ppl_g64_64seq.json` | 11.893 | 11.893 | PASS |
| P4 | g64 PPL gptq 12.1679 | `ppl_g64_64seq.json` | 12.1679 | 12.1679 | PASS |
| P5 | g128 PPL nac 12.1644 < gptq 12.5032 < minmax 12.5436 | `ppl_g128_64seq.json` | 12.1644 | 12.1644 | PASS |
| T1 | NAC vs minmax g64 MMLU p=3.285e-9 | `stats_pairs.json` | 3.285e-09 | 3.285e-09 | PASS |
| T2 | NAC vs minmax g128 MMLU +6.71pp p=7.895e-9 | `stats_pairs.json` | 7.895e-09 | 7.895e-09 | PASS |
| T3 | NAC-v2 vs NAC MMLU p=2.138e-5 | `stats_pairs.json` | 2.138e-05 | 2.138e-05 | PASS |
| T4 | NAC-v2 vs GPTQ MMLU 平手 p=0.7161 | `stats_pairs.json` | 0.7161 | 0.7161 | PASS |
| T5 | NAC-v2 vs NAC GSM8K p=0.01242 | `stats_pairs.json` | 0.01242 | 0.01242 | PASS |
| T6 | NAC-v2 vs GPTQ GSM8K p=1.225e-4 | `stats_pairs.json` | 0.0001225 | 0.0001225 | PASS |

**33/33 PASS**
