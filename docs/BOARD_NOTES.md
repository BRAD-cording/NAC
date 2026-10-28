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

## 2026-10-27 R3 全链收官：双事故 + 判别分 + 叙事级结论
- **全链**：x86 原生 toolkit 1.2.3 构建 07:21 DONE → `models/local_w4a16_plain.rkllm`（1,410,933,676B，md5 `12aaf476a8f95eb5e8f4699d8e6b4256`）→ scp 回传板上 `models/local_w4a16_plain_x86.rkllm`（md5 一致）→ 板端评测出分。
- **事故①（wd 提前触发）**：`r3_x86_watcher.sh` 原以">1GB 即视为传完"为判据，1.41GB 模型传到 ~70% 就被误判，评测拉起读半截模型 → `rkllm_init failed: -1` 全灭。**修复**：改为「>1GB 且大小连续 3 轮（~3min）不再增长」才触发。
- **事故②（LD_LIBRARY_PATH 污染）**：Shell 工具环境 `LD_LIBRARY_PATH` 指向 Trae server 自带老库（缺 GLIBCXX_3.4.26），评测/客户端 dlopen 秒退。**修复**：`env -u LD_LIBRARY_PATH` 重拉，smoke PASS。教训：容器/host 环境变量对 Shell 工具子进程同样污染，拉起长任务前先 `env -u`。
- **R3 判别分（[acc_gsm8k_qwen1.5b_w4a16plain_x86_zs.json](../results/acc_gsm8k_qwen1.5b_w4a16plain_x86_zs.json) / [acc_mmlu_qwen1.5b_w4a16plain_x86_zs.json](../results/acc_mmlu_qwen1.5b_w4a16plain_x86_zs.json)）**：
  | 模型 | GSM8K zs 200 | MMLU zs 969 |
  |---|---|---|
  | 本地 plain build（g64） | **0.015（崩）** | 0.4974（存活） |
  | 厂商预转基线 | 0.5550 | 0.5583 |
- **叙事级结论（≈0.005 分支确认）**：本地构建的崩坏模式与 10-23 事故前"注入构建"板端行为**完全同型**（GSM8K 崩 / MMLU 存活）→ 10-23 的板端崩塌有了更简单解释：**崩在本地 toolkit 构建路径共性，与 extra_qparams 注入/float-zp 重解释无关**。问题 B（NPU zp 重解释）嫌疑整体降级，outline H2 必须重写；MMLU 0.4974 vs 0.5583 的 6pp 差亦待归因。**下一步 bisect**：本地 plain build vs 厂商预转的构建差异（tokenizer/对话模板、LLM_HEAD/embed 处理、校准 dataset、quant 参数），产出差异假设清单 + R4 最小对照实验。

## 2026-10-27 凌晨：R4-a 构建连环事故与修复（OOM/转义/判活三连）
- **根因①（OOM）**：R4-a 512 条多域数据集构建必死——dmesg 实锤 `oom_kill_process python3.10 anon-rss:12.68GB`（.wslconfig memory=13GB，实际可用 ~12GB；swap 20GB 但 OOM 时仅用 5MB = 分配突发快于 kswapd 换出）。**条数减半不解决**：128 条仍爆；**R3 同 13GB 上限 128 条 wikitext 能过** → 多域内容（中文/代码/数学）使校准内存峰值显著高于等量英文 wikitext，与条数弱相关。**修复：`sysctl vm.swappiness=100`**（root 写 /etc/sysctl.d/99-swapaggro.conf 持久化）→ swap 主动承接 3GB、物理内存回落 9-11GB，构建稳定过 Optimizing 1/28。宿主 16GB 不敢调大 VM 上限（Windows 只剩 2-3GB）。
- **根因②（keeper 转义地狱）**：r4a_keeper.sh 原判活查询为 ssh 双引号内嵌 bash -c 单引号内嵌 \$p 多层转义 → 解析崩坏返回空 → alive 恒空 → **每 60-90s 误判 DEAD 疯狂 schtasks restart**（01:34-01:45 日志 11 轮）。**修复：判活逻辑抽成远端脚本** ~/r3x86/check_r4a.sh（本地副本 scripts/check_r4a.sh），keeper 只调 `$SSH "wsl -d Ubuntu -- bash .../check_r4a.sh"`，彻底消除转义层。教训：**跨机判活一律远端脚本化，不做多层内联转义**。
- **根因③（/end 副作用）**：schtasks /end 杀 wsl.exe 前台树 → WSL 内 python 成孤儿后仍可能被会话回收；且 .bat PID 守卫读旧 PID 残影会 exit 0 拒绝新实例。诊断期手动操作只用 /run 不用 /end。
- **cmd 重定向坑（新）**：ssh 到 Windows 后命令经 cmd.exe 解析，`bash -c '... > file'` 的 `>` 被当 cmd 重定向（报 "The system cannot find the path specified"）→ 必须用双引号嵌套 `bash -c \"...\"`（cmd 双引号内不解析 `>`）。
- **传输链补充**：stdin 管道过 ssh→wsl.exe 不可用（cat 收不到）；scp 单文件 231KB 实测未触发 204800B 截断（之前结论仅部分场景成立）；base64 -d 解码远端文件后必须 md5 双端校验。
- **R4-a 最终配置**：校准集缩为 **64 条多域**（gsm16/code8/zh16/wt24，data/quant_dataset_r4a64.json，md5 9077d495…；512 条原件与 128 条版均留档 data/）。对 B1（校准数据分布假设）检验效力不变：R3=128 条英文 wikitext vs R4-a=64 条多域，分布差异仍是唯一主变量；条数缩减对 normal 算法校准统计影响小（128 条已属保守量级）。
- **回传事故（scp 截断确定性复现）**：r4a-watch 的 scp 拉取 1.41GB 模型 3 次全部截断在 204800B 且 md5 完全相同（e6e91088…，确定性 bug 非偶发）→ FATAL ABORT。**修复：改用 sftp**（`sshpass sftp` 批处理 `get`）全量拉取成功（md5 1494481a… 与源一致，~2h @1.8MB/s）。教训：**Windows sshd 的 scp 拉方向不可信，大文件一律 sftp**。swappiness=100 后 R4-a 构建 28 层校准 + export 全程稳定（05:30 DONE）。

## 2026-10-27 上午：R4-a 判别分 → B1 证伪 → R4-b（opt=3）启动
- **R4-a 判别分**：smoke PASS（9s）；**GSM8K zs 200 = 0.0000**（21min，acc_gsm8k_qwen1.5b_r4a_zs.json）；MMLU zs 969 在跑。
- **B1（校准数据分布）证伪**：R3（128 条 wikitext）0.015 vs R4-a（64 条多域）0.0000——换多域数据无任何恢复，崩坏模式不变。**R3 vs R4-a 字节 diff 地图**（scripts/rkllm_diffmap.py，64KiB 块）：3 区共 46.4%（624MB），全部落在 448–1123MiB 量化层权重区，与 R3 vs vendor 差异区域同型 → **校准数据确实改变权重字节但完全不改变板端行为** → 崩坏与校准数据无关实锤。
- **R4-b 启动（B3：optimization_level 1→3）**：脚本推为笔记本 r4a_build.py（备份 r4a_build_opt1.py.bak；PID/log/标记前缀复用 r4a 生态，keeper/watcher 零改动），数据集保持 64 条多域（单变量）；R4-a 模型笔记本侧备份 local_w4a16_r4a_opt1ds64.bak.rkllm。08:20 起跑 Optimizing，ETA ~14:30。watcher 已改 sftp 回传 + r4b_pipeline.sh 自动评测。构建期笔记本睡眠再次禁用。
- **候选假设剩余**：B3（opt level，R4-b 在跑）→ B2（quantized_algorithm）→ **B7（新，升位）：num_npu_core=2 或其他 build 参数与 vendor 不一致**——vendor model card 未披露 opt level/algorithm/npu core，本地全部显式传参，任何一个都可能是崩因。

## 2026-10-27 下午/晚：R4-b 判别分 → B3 证伪 → 签名对齐 → R4-c（npu_core=1）
- **R4-b 判别分**：GSM8K zs 200 = **0.025**（opt 1→3 无实质变化，0.015→0.025 噪声级）；MMLU = **0.5191**（略高于 R3/R4a 的 0.4974——opt=3 图优化对单步任务小改善，核心崩坏不动）→ **B3 证伪**。R4-b 模型 1.294GB（比 R3/R4a 的 1.411GB 小 8.3%，opt=3 算子融合改变结构；vendor 1.41GB 同 R3 → vendor 大概率非 opt=3）。
- **toolkit 1.2.3 签名对齐（内省实锤）**：`build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8', quantized_algorithm='normal', target_platform='rk3588', num_npu_core=1, extra_qparams=None, dataset=None, hybrid_rate=0.0, max_context=4096)`。对照本地调用：**唯一显式偏离默认且 vendor 未必使用的参数 = num_npu_core=2**（默认 1）；w4a16/rk3576 是 vendor 必同项；max_context 默认 4096 与 vendor 一致。→ **B7 焦点收窄至 num_npu_core**。
- **R4-c 启动（B7）**：与 R3 完全一致仅 npu_core=1，数据集回 R3 原版 128 条 wikitext（严格单变量）。21:20 起跑 Optimizing，ETA ~00:40 出模型 → watcher 自动 sftp 回传 + r4c_pipeline 自动评测，判别分 ~01:30。恢复则 npu_core=2 是元凶（论文叙事再修正）；仍崩则 npu_core 排除，转 B2（algorithm）/target_platform。
- **传输链提速**：R4-b 模型 sftp 回传仅 ~20 分钟（12:04 OK，1.29GB）——R4-a 首次拉取的 2h 可能含 Windows cp 中转等待；现流程（cp→sync→sftp）已顺。watcher 为 transient 单元（systemd-run），完成即消失，需 systemd-run 重拉（10-27 深夜踩点）。

## 2026-10-28 凌晨/上午：R4-c 判别 → B7 证伪 → 算法列表泄露 → R4-d（gdq）
- **R4-c 判别分（03:52 ALL DONE，全链自动）**：GSM8K = **0.015** / MMLU = **0.4974**——与 R3 **逐位一致** → **B7（npu_core）证伪**。模型 1.391GB。
- **字节侦察两个关键事实**：(a) R4c vs vendor 头部 0–64KiB 也不同（R3/R4a vs vendor 头部 448MiB 全同）→ npu_core 改头部指纹，vendor 头部与 R3（npu2）一致 → vendor 也是 npu2 类配置；(b) **vendor 与 R3 文件总大小逐字节相同（1,410,933,676B）→ 若 vendor 用 hybrid_rate>0（部分层 w8a8）大小必变 → hybrid 假设排除**（B8）。
- **toolkit 1.2.3 算法列表泄露（rkllm/api/rkllm.py py 源码 docstring，.so 无 docstring）**：`quantized_algorithm options: [none, normal, normal_r1~normal_r64, gdq, gdq_r1~gdq_r64, grq, grq_r1~grq_r64]`，gdq/grq 仅支持 4bit（w4a16 ✓），且 extra_qparams 机制标注 "cached by gdq/grq [gdq/grq.qparams]" → 高级算法可产生与 normal 完全不同的 scale/zp 决策，同 dtype 同打包尺寸不变——**恰好同时解释"同头部同尺寸同结构但权重区 payload 全不同 + 行为天差地别"**。
- **R4-d 启动（B2）**：algorithm normal→gdq，其余与 R3 全同（opt1/w4a16/rk3576/npu2/R3 数据集）。构建已 ALIVE，判别分 ~+5h。恢复 → B2 证实（vendor 用 gdq 类高级算法，"toolkit 默认参数不能开箱复现 vendor 质量"本身就是论文可写发现）；仍崩 → 试 grq / normal_r 变体 / 数据条数。
- **假设战况总览**：B1 数据 ✗ / B3 opt ✗ / B5 元数据 ✗ / B6 模型源 ✗ / B7 npu_core ✗ / B8 hybrid ✗ / B4 版本降级 → **剩 B2（algorithm 家族）与 target_platform（弱）**。
- **R4-d 启动真相勘误（07:20 复盘）**：前次会话记录的"R4-d 已启动（ALIVE / Optimizing 0/28）"**未生效**——build_r4a.log 第 110 行（R4-c DONE）之后无任何运行记录，04:00-07:13 板上无构建进程（keeper 因尾部 DONE 标记正确不重启）。**实际首次运行由 30 分钟级"构建监控"定时自动化于 07:13:03 恢复触发**（schtasks Last Run 实锤），07:15:12 该自动化又重拉了 r4a-watch（transient 单元）。当前进程 PID 56653，07:13:45 起跑，RSS ~10GB，gdq 算法 Optimizing 中。**教训**：(a) 跨会话交接的"已启动"断言必须用日志行计数验证（`grep -c 'load ret='`），不能只看 ALIVE；(b) 监控自动化具备真实恢复能力，其消息口径已同步更新为 R4-d（含恢复动作边界：仅 DEAD 且无 DONE 时才 schtasks 重启）。

## 2026-10-28 晚：R4-d 判别 → B2 主形态（gdq）证伪
- **R4-d 判别分（19:54 ALL DONE，全链自动，构建 07:13→19:05 ≈11.9h，gdq 实测 ~1500s/层）**：GSM8K zs 200 = **0.01**（崩坏带纹丝不动，vendor 0.5550）/ MMLU zs 969 = **0.5387**（本地系列最高：R3 0.4974 → R4a 0.4974 → R4b 0.5191 → R4d 0.5387，与 vendor 0.5583 差距收窄到 2pp）→ **B2 主形态 gdq 证伪**：高级算法确实改变模型行为（MMLU +4pp），但 GSM8K 崩坏与算法无关。
- **模型特征**：local_w4a16_r4d.rkllm = 1,410,933,676B（与 vendor/R3 逐字节同大小——gdq 同 dtype 同打包的预测兑现）；构建产物 gdq.qparams 1.32GB（extra_qparams 缓存机制，docstring "cached by gdq/grq" 的实证）。
- **模式学新观察（待纳入归因）**：全部本地构建呈现"**MMLU（短生成判别）存活 ≈0.50-0.54，GSM8K（多步长链生成）崩 ≈0.01-0.025**"二相行为 → 崩坏可能不在权重全局质量，而与**长链解码/数值放大**相关（疑点：lm_head/LLM_HEAD 处理、ctx/解码链路），或算术误差逐 token 放大。vendor 同评测框架 0.5550 → 评测面无问题。
- **假设战况总览（更新）**：B1✗ B3✗ B5✗ B6✗ B7✗ B8✗ **B2-gdq✗** → 剩：**B2 其他形态（grq / normal_r1~r64 / gdq_r1~r64）**、target_platform（弱）、B4 版本（降级未排除）。
- **R4-e 决策（23:58）**：R4-e 此前从未启动（跨会话误记，笔记本 DEAD 且日志尾为 R4-d DONE 实锤）。候选项：grq（~13h）/ normal_r1（~6h）/ 转 target_platform——待用户拍板。
