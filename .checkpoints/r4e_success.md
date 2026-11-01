# CCG checkpoint — 2026-11-01 02:38 (逢十轮)

## 六要素
1. **功能名**: NAC 论文 Q1 R4-e(grq) 板端研发管线
2. **生效条件**: WSL ~/r3x86 grq 构建完成; 修复自 _linlr_shim.py(LinearLR verbose neutralized), 上坏实例于 26/28 崩溃
3. **子功能**: (a) sim 截断重跑 5 种; (b) grq build→save_qparams→export→回传→512 evalset; (c) Finding 1 闸门改写
4. **执行**: Optimizing 28/28 无崩, [r4a] build ret=0, export OK models/local_w4a16_r4e.rkllm (1410933676B), DONE; 回传 md5=90049d00... 匹配, 1.4G 落地; r4e-eval.service active (r4e_zs 512 eval 200题). sim minmax partial: acc 0.3839 / acc_answeris 0.3929
5. **验证**: 以 tqdm elapsed 帧 + CPU TIME+ 为准(ps STARTED/ELAPSED 曾矛盾); 越 26/28 崩溃点 = 根治实证
6. **不适用**: OpenViking MCP 断开, Hindsight 无 apiToken(401) —— 此归档落盘 .checkpoints/ 后备, 不入 cg 记忆库
