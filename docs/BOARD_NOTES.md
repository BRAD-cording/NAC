# RK3576 板上踩坑记录（D1–D2 实测，2026-10-06）

## 环境
- 本机即 NanoPi RK3576（M5，aarch64，16GB，NVMe 26G 根分区剩余 ~16G）
- Python 3.11.2；venv 在 nac_paper/.venv；torch 2.14.1+cpu（aarch64 用 pytorch cpu index）
- pip 从 pypi.org 下载极慢 → 用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- HF 下载用 `HF_ENDPOINT=https://hf-mirror.com`

## RKLLM runtime / 模型
- 本地 rknn-llm 仓库是 v1.1.4，但社区预转 .rkllm 多为 toolkit 1.2.3 → 提取
  `release-v1.2.3` tag 的 .so 到 `rkllm-runtime-1.2.3/Linux/librkllm_api/aarch64/`
  （tag 名带 `release-` 前缀，非 `v1.2.3`；.so 非 LFS，git show 可直接取）
- 驱动 rknpu 0.9.8 为 DRM 风格：无 /dev/rknpu，走 /dev/dri/renderD129（用户需在 render 组，已满足）
- 用模型：HanzoHuang/Qwen2.5-1.5B-Instruct-RKLLM → RK3576/Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm
  （1.41GB，SHA256 70b4b3c5…24e332c 已验证；同仓库另有 w8a8 与 RK3588 版本；3B 可用 VRxiaojie/Qwen2.5-3B-Instruct-RKLLM1.1.4，但它是 1.1.4 toolkit 转的，注意版本匹配）

## ctypes 绑定（llm_client.py）关键教训：1.2.3 vs 1.1.4 结构体差异
1. RKLLMParam 在 top_k 后**新增 n_keep(int32)** → 全体字段偏移 +4，漏掉会 segfault（init 阶段）
2. RKLLMInput 在 input_type 前**新增 role(char*) + enable_thinking(bool)** → 否则报
   "The input_type of rkllm_input is not set"
3. RKLLMExtendParam：embed_flash/enabled_cpus_num/enabled_cpus_mask/n_batch/use_cross_attn
   + reserved[104]（总 120B）
4. RKLLMInferParam 尾部**新增 keep_history(int32)**（0 = 单轮，评测必须设 0）
5. RKLLMResult：token_id 后新增 logits(16B) + perf(20B)，last_hidden 移到尾部
6. 回调签名从 void 改为 **int**，必须 return 0（非 0 会暂停推理）
7. **最佳实践：调用 `rkllm_createDefaultParam()` 让库自己构造 param**，再覆盖字段，
   规避布局风险（restype=RKLLMParam）

## 初始化参数坑
- `enabled_cpus_num=0` 会报 "number of enabled CPUs must be >= number of NPU cores"（RK3576 NPU 2 核）
- RK3576 大小核：**cpu0-3 = A53@2.016GHz，cpu4-7 = A72@2.2GHz**（policy0/policy4 实测 max freq + A53 PMU 先注册确认）
  → enabled_cpus_mask=0xF0（4×A72 大核）
- embed_flash=1（官方 demo 默认）

## 数据集
- datasets v5 不再认旧名 `wikitext` → 用 `Salesforce/wikitext`，"wikitext-2-raw-v1"

## 板上功耗采样
- /sys/class/power_supply/simple-vin 只有电压(~4.9V)无电流；无 powercap → 整机功率需外接 INA260/UM34C

## 实测性能（Qwen2.5-1.5B W4A16，定频，4×A72）
- init ~3.0s；TTFT ~100ms@128tok / ~144ms@512tok；decode ~11 tok/s（n_batch=1, embed_flash=1）
- 定频路径：sudo python -m src.utils.freq --mode performance（CPU performance + NPU 950MHz + DDR 2736MHz）

## 2026-10-07 3B 模型与 zero-shot 协议（T2）

- 3B 预转模型下载完成：`models/Qwen2.5-3B-Instruct_RK3576_w4a16.rkllm`
  （2,389,038,316 B；sha256 5f2480e1...6908 见同名 .sha256；远端总大小即此值，
  hf-mirror curl Range 超界返回 416，"断点续传"追加的是 416 错误页垃圾，需 truncate）
- 冒烟 PASS：init 6.4s，decode ~6.2-7.0 tok/s（与 NAC 校准并行时），生成质量正常
- **关键发现：3B 预转模型 5-shot MMLU 崩坏（acc=0.10@50题 < 随机 0.25）**：
  长 few-shot prompt（~430 chars）下模型复述 few-shot 示例而非回答真实问题
  （复现：FEWSHOT+题 → 输出 "Let's address each question... Which organelle produces ATP?
  The correct answer is B"；无 few-shot → 直接正确输出 "A. luster"）。
  疑为预转 W4A16 长上下文 prefill 退化。1.5B 同 prompt 正常（acc=0.4469）。
- 处置：accuracy.py 加 `--fewshot-num`（0=zero-shot，MMLU zero-shot prompt
  带 "Answer with a single letter" 指令）。3B 用 zero-shot；1.5B 需补跑同协议
  zero-shot 对照（acc_mmlu_qwen1.5b_w4a16_zs.json）保证可比性。
- NAC grid 校准修复两处 bug 后顺利运行：
  1) linear pre-hook 需 `register_forward_pre_hook(..., with_kwargs=True)`
  2) 传播阶段直接调用 decoder block 需显式传 `position_embeddings`
     （`model.model.rotary_emb(x, pos_ids)`，新版 transformers 由顶层下传）
  节奏 ~40-70s/块（与评测并行时），ratio_mean 稳定在 0.4
- 后台作业教训：`cd X && setsid ... &` 中 & 使 cd 只在后台子 shell 生效，
  前台同命令行的后续相对路径用原 cwd → 日志检查一律用绝对路径

## 2026-10-07 凌晨：zero-shot 评测结果 + torch CPU 吞吐教训

### 板上 zero-shot 精度（T2 完成，同协议可比）
| 模型 | MMLU zs | GSM8K zs | 备注 |
|---|---|---|---|
| 1.5B W4A16 | 0.5583 (969题) | 0.5550 (200题) | 5-shot MMLU 为 0.4469 |
| 3B W4A16 | 0.6037 (969题) | 0.6700 (200题) | 5-shot 崩坏 0.10（复述示例） |
- GSM8K 用 --max-new-tokens 512（32 会截断推理）；3B ~45s/题，1.5B ~25s/题

### torch 2.14.1 aarch64 CPU 吞吐（重要）
- **bf16 F.linear 病态慢**：mkldnn bf16 路径不支持 A55/A72 → 回退非 BLAS 核，
  [16,1023,1536]×[1536,8960] 单次 >30s（<15 GFLOPS 且极不稳定）
- **fp32 sgemm 也仅 ~10 GFLOPS**（4×A53，基准实测 41.8s/次同形状）
  → 该 wheel 的 BLAS 非优化实现，PPL 验证类任务吞吐被锁死在 ~10-15 GFLOPS
- eval_ppl bs=1 时纯带宽受限（每 token 重读全模型权重），bs=16 提升有限
- **对策：validate 类任务全部放 A72（taskset -c 4-7）**：g64 校准 19s/块
  （A53 时 53-57s/块，3×）；A72 上 validate 单 pass ~55-60min
- 微基准教训：heredoc python 的 pkill 匹配不到（cmdline 已 exec 成 python），
  用 ps+kill -9 清理，防多基准抢核污染测量
- LD_PRELOAD 系统 OpenBLAS（0.3.21）失败：torch 调 `sbgemm_`（bf16 gemm 扩展），
  bundled libopenblas 有而 0.3.21 没有 → ImportError，需 shim 不可行；
- 源码编译 OpenBLAS 0.3.29（ARMV8+LAPACK+BUILD_BFLOAT16）后 LD_PRELOAD：
  torch fp32 linear 吞吐完全不变（41.6s/41.8s，~10.7 GFLOPS）→ torch 走
  mkldnn 自带核，动态链接劫持无效，彻底放弃；
- **最终定论：本板 torch CPU 前向吞吐 ~10 GFLOPS（A53/A72 同），
  16×1024 单 pass ~1.6-2h；g128（4-7）与 g64（0-3）并行跑互相让核**

## 后续记录索引（2026-10-18 补）
10-08 起板端工程记录（extra_qparams 注入链路、qemu 构建病理、qparams 禁落 /tmp、重启自愈编排、E2/E3/E4 自动链）统一并入 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) §4 与进展日志，本文件保留 D1–D7 期踩坑正文，不再续写。

## 2026-10-26 R3 笔记本构建静默死亡与防复发（infra 踩坑）
- **事故**：build 9（x86 plain build）于 10-25 23:59 前后死——Windows AC 睡眠超时 0x0e10（3600s），nohup 后台构建不算用户活动 → 笔记本睡眠 → WSL 整体重启，进程蒸发（dmesg 无 OOM、日志无 FAIL 标记，守望 shell 未触发通知）。日志停在 Optimizing 24/28 [5:01:42]，损失 ~5h。
- **处置**：崩溃日志备份 `~/r3x86/build_x86.crash1026.log`；`start_build.sh` 重启（pid 117390，01:45 起跑，ETA：Optimizing ~6h + export ~45min ≈ 08:30 模型落盘，传输+板评 ~3h → 中午出分）；**远程 powercfg：AC 睡眠/休眠超时清零 + 合盖不动作**（原值 0x0e10，构建结束后恢复 `powercfg /change standby-timeout-ac 3600`）。
- **教训**：跨机长任务必须 (a) 禁目标机睡眠 (b) 守望判据加"日志 mtime 超过 N 分钟未更新即报警"，不能只等 DONE/FAIL 关键词（进程蒸发不产生任何标记）。
