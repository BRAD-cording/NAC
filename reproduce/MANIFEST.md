# Artifact manifest

Every `.rkllm` used in this study, with md5 and the runtime dtype reported by librkllm.
Artifacts with identical md5 are the *same file* despite different names — see the NAC/E3 rows.

| artifact | size (B) | md5 | dtype | role | evals |
|---|---|---|---|---|---|
| `Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm` | 1,410,933,676 | `90330bf182cd5230…` | W4A16 | vendor reference (1.5B) | gsm8k_qwen1.5b_w4a16_zs: GSM8K 0.555; mmlu_firsttok_qwen1.5b_w4a16: MMLU 0.4066; gsm8k_qwe |
| `asym_minmax_w4a16.rkllm` | 1,410,933,676 | `6de5a332bbb7406a…` | W4A16 | asymmetric minmax arm, W4A16 | gsm8k_qwen1.5b_w4a16asymminmax_zs: GSM8K 0.01 |
| `local_w4a16_plain_x86.rkllm` | 1,410,933,676 | `12aaf476a8f95eb5…` | W4A16 | local rebuild, normal/opt1, W4A16, x86 native | mmlu_qwen1.5b_w4a16plain_x86_zs: MMLU 0.4974; gsm8k_qwen1.5b_w4a16plain_x86_zs: GSM8K 0.01 |
| `local_w4a16_r4a.rkllm` | 1,410,933,676 | `1494481a2fad42ce…` | W4A16 | local rebuild, normal/opt1, multi-domain calibration | mmlu_qwen1.5b_r4a_zs: MMLU 0.4974; gsm8k_qwen1.5b_r4a_zs: GSM8K 0.0 |
| `local_w4a16_r4b.rkllm` | 1,294,257,260 | `0005f15e9ca4d1d6…` | W4A16 | local rebuild, normal/opt3 | mmlu_qwen1.5b_r4b_zs: MMLU 0.5191; gsm8k_qwen1.5b_r4b_zs: GSM8K 0.025 |
| `local_w4a16_r4c.rkllm` | 1,391,642,196 | `1f067442d1890ca8…` | W4A16 | local rebuild, normal/opt1, num_npu_core=1 | mmlu_qwen1.5b_r4c_zs: MMLU 0.4974; gsm8k_qwen1.5b_r4c_zs: GSM8K 0.015 |
| `local_w4a16_r4d.rkllm` | 1,410,933,676 | `f06e53f8748d3b60…` | W4A16 | local rebuild, gdq, W4A16 | gsm8k_qwen1.5b_r4d_zs: GSM8K 0.01; mmlu_qwen1.5b_r4d_zs: MMLU 0.5387 |
| `nac_qwen1.5b_g64_mmctrl.rkllm` | 1,653,561,316 | `3bd8c0ba73f056a9…` | W4A16_G64 | E3 control arm — same bytes as NAC arm (3bd8c0ba) | gsm8k_qwen1.5b_g64mmctrl_zs: GSM8K 0.005; mmlu_qwen1.5b_g64mmctrl_zs: MMLU 0.5614 |
| `nac_qwen1.5b_g64_nac.rkllm` | 1,653,561,316 | `3bd8c0ba73f056a9…` | W4A16_G64 | NAC-v1 arm — injection never applied (md5 == 3bd8c0ba) | gsm8k_qwen1.5b_g64nac_zs: GSM8K 0.005; gsm8k_qwen1.5b_g64nac: GSM8K 0.12; mmlu_qwen1.5b_g6 |
| `nacv2_qwen1.5b_g64_intzp.rkllm` | 1,653,561,316 | `3bd8c0ba73f056a9…` | W4A16_G64 | NAC-v2 arm — same bytes as NAC arm (3bd8c0ba) | gsm8k_qwen1.5b_g64_nacv2intzp_zs: GSM8K 0.005; mmlu_qwen1.5b_g64_nacv2intzp_zs: MMLU 0.561 |
| `sym_minmax_g64.rkllm` | 1,653,561,316 | `3bd8c0ba73f056a9…` | ? | minmax g64 arm — same bytes as 3bd8c0ba |  |
| `sym_minmax_g64_v2.rkllm` | 1,653,561,316 | `d6ff154dc261236f…` | W4A16_G64 | minmax g64 arm — distinct bytes (d6ff15) | gsm8k_qwen1.5b_g64symminmax_zs: GSM8K 0.005 |
| `tiny_inject.rkllm` | 47,901,980 | `db6492845c58fdfc…` | ? |  |  |
| `tiny_inject_s.rkllm` | 47,901,980 | `1627694b65ca2120…` | ? |  |  |
| `tiny_inject_w.rkllm` | 47,901,980 | `0c7f2cd990f56a79…` | ? |  |  |
| `tiny_range_empty.rkllm` | 47,901,980 | `9da41ae7db15c5cf…` | ? |  |  |
| `tiny_range_sym7.rkllm` | 47,901,980 | `9da41ae7db15c5cf…` | ? |  |  |
| `tiny_range_sym8.rkllm` | 47,901,980 | `67644266e8769083…` | ? |  |  |
