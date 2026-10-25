"""AWQ 基线校准（P0-1 扩展：Lin et al. 2023，项目内实现）。

忠实实现 AWQ 核心（激活感知 scale + 前置折叠 + clip 网格），量化框架与
NAC/GPTQ 完全一致以保证公平对比：
- 同一校准数据（configs/nac.yaml：wikitext-2 val，capture 8×512 fp32 CPU）
- 同一块级顺序传播（折叠+量化写回活体模型 → 下一块真实输入）
- 同一 qparams 产物格式 {block: {name: {scale, zp}}}，整值 zp（NPU 公式
  W=scale*(q-zp) 整格点域，与 GPTQ 基线及问题 B 处置对齐）
- 折叠产物 .fold.pt 只存 scale 向量 {s_attn, s_o, s_mlp, s_down}（每块
  ~1.2 万浮点）；eval --fold 用 src.nac.awq.fold_block_ 重建折叠权重，
  与本校准位级一致（不落 6GB 权重，避开板端 RAM 峰值）

用法（板上 A72）：
  taskset -c 4-7 .venv/bin/python scripts/awq_calibrate.py \
      --model models/Qwen2.5-1.5B-Instruct-hf --group-size 64 \
      --out results/awq_qwen1.5b_g64
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nac.awq import (ALPHA_GRID, HEAD_DIM, MAXSHRINK, CLIP_GRID,  # noqa: E402
                         _grid_search, awq_quant_weight, chan_weights,
                         expand_s_o, fold_block_)
from src.nac.quantize import dequantize_weight, quantize_weight  # noqa: E402
from src.utils.data import load_calib_samples  # noqa: E402

CAPTURE_NAMES = ("q_proj", "o_proj", "gate_proj", "down_proj")  # 组代表（共享输入）


def main():
    ap = argparse.ArgumentParser(description="AWQ baseline calibration (CPU)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "nac.yaml"))
    ap.add_argument("--out", required=True, help="输出前缀，如 results/awq_qwen1.5b_g64")
    ap.add_argument("--limit-blocks", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["nac"]
    gs = args.group_size
    torch.set_num_threads(4)
    print(f"[awq] loading {args.model} (fp32) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    samples = load_calib_samples(cfg["calib_dataset"], cfg["num_samples"], tokenizer, cfg["seqlen"])
    input_ids = torch.cat(samples, dim=0)[: cfg["capture_samples"]]
    print(f"[awq] calib seqs={input_ids.shape[0]} seqlen={cfg['seqlen']} group={gs}", flush=True)

    blocks = model.model.layers
    n_blocks = len(blocks) if args.limit_blocks is None else args.limit_blocks

    # 1) FP 前向捕获块输入（顺序传播用）
    block_inputs: list[torch.Tensor] = []
    hooks = []

    def block_pre_hook(_idx):
        def hook(_mod, aargs, _kw):
            block_inputs.append(aargs[0].detach().float().cpu())
        return hook

    for i, b in enumerate(blocks[:n_blocks]):
        hooks.append(b.register_forward_pre_hook(block_pre_hook(i), with_kwargs=True))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
    print(f"[awq] captured {len(block_inputs)} block inputs", flush=True)

    # 2) 逐块：块前向捕获 Linear 输入激活 → 搜 s → 折叠 → 量化写回 → 传播
    qparams: dict[int, dict] = {}
    fold: dict[int, dict] = {}
    t0 = time.time()
    pbar = tqdm(range(n_blocks), desc="AWQ blocks")
    for bi in pbar:
        x = block_inputs[bi]
        acc: dict[str, torch.Tensor] = {}
        cnt: dict[str, int] = {}
        hooks = []

        def linear_pre_hook(name):
            def hook(_mod, aargs, _kw):
                xx = aargs[0].detach().reshape(-1, aargs[0].shape[-1]).float()
                if name not in acc:
                    acc[name] = torch.zeros(xx.shape[1])
                    cnt[name] = 0
                acc[name] += xx.abs().sum(0)
                cnt[name] += xx.shape[0]
            return hook

        for name, mod in blocks[bi].named_modules():
            if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in CAPTURE_NAMES:
                # hook 以模块后缀名计数（q/o/gate/down），与下方 x_bar[...] 后缀查询对齐
                hooks.append(mod.register_forward_pre_hook(
                    linear_pre_hook(name.split(".")[-1]), with_kwargs=True))
        with torch.no_grad():
            pos_ids = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
            pos_emb = model.model.rotary_emb(x, pos_ids)
            blocks[bi](x, position_embeddings=pos_emb)
        for h in hooks:
            h.remove()
        x_bar = {k: acc[k] / cnt[k] for k in acc}  # 逐通道 mean|x|（q 覆盖 q/k/v，gate 覆盖 gate/up）

        blk = blocks[bi]
        attn, mlp = blk.self_attn, blk.mlp
        ws_attn = [attn.q_proj.weight.data.float(), attn.k_proj.weight.data.float(),
                   attn.v_proj.weight.data.float()]
        n_kv_heads = attn.v_proj.weight.shape[0] // HEAD_DIM
        gpd = attn.o_proj.weight.shape[1] // (n_kv_heads * HEAD_DIM)

        s_attn, a_attn = _grid_search(
            x_bar["q_proj"], torch.stack([w.abs().amax(0) for w in ws_attn]).mean(0),
            ws_attn, gs)
        s_o, a_o = _grid_search(x_bar["o_proj"], attn.o_proj.weight.data.float().abs().amax(0),
                                [attn.o_proj.weight.data.float()], gs,
                                orbit=(n_kv_heads, HEAD_DIM, gpd))
        s_mlp, a_mlp = _grid_search(
            x_bar["gate_proj"],
            torch.stack([mlp.gate_proj.weight.data.float().abs().amax(0),
                         mlp.up_proj.weight.data.float().abs().amax(0)]).mean(0),
            [mlp.gate_proj.weight.data.float(), mlp.up_proj.weight.data.float()], gs)
        s_down, a_down = _grid_search(x_bar["down_proj"],
                                      mlp.down_proj.weight.data.float().abs().amax(0),
                                      [mlp.down_proj.weight.data.float()], gs)

        fold_block_(blk, s_attn, s_o, s_mlp, s_down)
        s_o_full = expand_s_o(s_o, n_kv_heads, HEAD_DIM, gpd)
        fold[bi] = {"s_attn": s_attn, "s_o": s_o, "s_mlp": s_mlp, "s_down": s_down,
                    "alpha": {"attn": a_attn, "o": a_o, "mlp": a_mlp, "down": a_down}}

        # 折叠后全 Linear 量化（先折叠后量化；chan_w=(x̄/s)² 对应各消费者输入）
        # 键名用块内全路径（self_attn.*/mlp.*），与 NAC/GPTQ qparams 及
        # apply_qparams/build 注入的生态约定一致
        plan = [(f"self_attn.{n}", getattr(attn, n), xk, sf) for n, xk, sf in
                [("q_proj", "q_proj", s_attn), ("k_proj", "q_proj", s_attn),
                 ("v_proj", "q_proj", s_attn), ("o_proj", "o_proj", s_o_full)]] + \
               [(f"mlp.{n}", getattr(mlp, n), xk, sf) for n, xk, sf in
                [("gate_proj", "gate_proj", s_mlp), ("up_proj", "gate_proj", s_mlp),
                 ("down_proj", "down_proj", s_down)]]
        qp_i: dict[str, dict] = {}
        for name, mod, xkey, s_full in plan:
            w = mod.weight.data.float()
            scale, zp = awq_quant_weight(w, gs, chan_weights(x_bar[xkey], s_full))
            res = quantize_weight(w, gs, scale, zp)
            mod.weight.data = dequantize_weight(res.q, res.scale, res.zp).to(mod.weight.dtype)
            qp_i[name] = {"scale": scale, "zp": zp}
        qparams[bi] = qp_i

        if bi + 1 < n_blocks:
            with torch.no_grad():
                block_inputs[bi + 1] = blocks[bi](
                    x, position_embeddings=pos_emb).detach().float().cpu()
        del x, acc, x_bar
        pbar.set_postfix(sec=round(time.time() - t0, 0))

    # 3) 导出（qparams 与 NAC/GPTQ 同格式；fold 为 tiny scale 向量）
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qparams, str(out) + ".qparams.pt")
    torch.save(fold, str(out) + ".fold.pt")
    meta = {"model": args.model, "method": "awq", "group_size": gs,
            "bits": 4, "asym": True, "int_zp": True,
            "alpha_grid": {"step": 0.05, "n": len(ALPHA_GRID)},
            "clip": {"maxshrink": MAXSHRINK, "grid": CLIP_GRID},
            "fold_format": "scale vectors {s_attn, s_o(kv-row), s_mlp, s_down}; rebuild via src.nac.awq.apply_fold",
            "calib": {"dataset": cfg["calib_dataset"], "num_samples": cfg["num_samples"],
                      "capture_samples": int(input_ids.shape[0]), "seqlen": cfg["seqlen"]},
            "blocks_calibrated": n_blocks, "device": "cpu",
            "seconds": round(time.time() - t0, 1)}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[awq] done in {meta['seconds']}s -> {out}.qparams.pt + {out}.fold.pt", flush=True)


if __name__ == "__main__":
    main()
