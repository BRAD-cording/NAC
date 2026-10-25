"""GPTQ 基线校准（论文 P0-1：SOTA PTQ 对比，Frantar et al. 2022）。

忠实实现 GPTQ 的 Hessian 误差补偿核心（非对称 uint4、per-group min-max、
maxshrink 网格），量化框架与 NAC 完全一致以保证公平对比：
- 同一校准数据（configs/nac.yaml：wikitext-2 val，capture 8×512 fp32 CPU）
- 同一块级顺序传播（量化输出写回活体模型 → 下一块真实输入）
- 同一 qparams 产物格式 {block: {name: {scale, zp}}}（zp 为整值浮点，
  与 Rockchip 反量化 W=scale*(q-zp) 语义兼容，可直接进 eval_mmlu_sim.py
  的 qparams=<file> 配置与 export 流程）
- 块内 layer-parallel（H 由同一次块前向捕获；官方 true-sequential 变体
  需 3× 块前向，板上成本不成比例，meta 中如实标注）

用法（板上 A72）：
  taskset -c 4-7 .venv/bin/python scripts/gptq_calibrate.py \
      --model models/Qwen2.5-1.5B-Instruct-hf --group-size 64 \
      --out results/gptq_qwen1.5b_g64
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
from src.nac.quantize import QMAX, QMIN  # noqa: E402
from src.utils.data import load_calib_samples  # noqa: E402

LINEAR_TYPES = (torch.nn.Linear,)
BLOCKSIZE = 128
PERCDAMP = 0.01
MAXSHRINK, GRID = 0.8, 100


def find_params_group(wg: torch.Tensor):
    """官方 GPTQ 非对称组参数（含 maxshrink 网格），wg: (out, gs) 当前组内列。

    返回 scale, zp 形状 (out,)；zp 为整值浮点（Rockchip 语义 scale*(q-zp)）。
    """
    wmin, wmax = wg.amin(-1), wg.amax(-1)
    best_scale = best_zp = best_err = None
    for i in range(GRID + 1):
        p = 1 - i / GRID * MAXSHRINK
        xmin2, xmax2 = p * wmin, p * wmax
        scale = (xmax2 - xmin2).clamp_min(1e-10) / QMAX
        zero = torch.round(-xmin2 / scale)  # 不截断：全正组 zp 可为负，靠 q 的 clamp 兜底（官方语义）
        q = torch.clamp(torch.round(wg / scale.unsqueeze(-1)) + zero.unsqueeze(-1),
                        QMIN, QMAX)
        err = (scale.unsqueeze(-1) * (q - zero.unsqueeze(-1)) - wg).pow(2).sum(-1)
        if best_err is None:
            best_scale, best_zp, best_err = scale, zero, err
        else:
            better = err < best_err
            best_scale = torch.where(better, scale, best_scale)
            best_zp = torch.where(better, zero, best_zp)
            best_err = torch.where(better, err, best_err)
    return best_scale, best_zp


def gptq_linear(W: torch.Tensor, H: torch.Tensor, gs: int):
    """GPTQ 核心：Hessian 逆的 Cholesky 列序误差补偿。

    W: (out, in) fp32 就地修改为反量化权重；H: (in, in) fp32。
    返回 (scale, zp) 形状 (out, in/gs)。
    """
    out_f, in_f = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = PERCDAMP * torch.mean(torch.diag(H))
    H[range(in_f), range(in_f)] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)

    ng = in_f // gs
    scale_all = torch.zeros(out_f, ng)
    zp_all = torch.zeros(out_f, ng)
    for i1 in range(0, in_f, BLOCKSIZE):
        i2 = min(i1 + BLOCKSIZE, in_f)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            col = i1 + i
            if col % gs == 0:
                sc, zp = find_params_group(W[:, col: col + gs])
                scale_all[:, col // gs] = sc
                zp_all[:, col // gs] = zp
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale_all[:, col // gs]) + zp_all[:, col // gs],
                            QMIN, QMAX)
            dq = scale_all[:, col // gs] * (q - zp_all[:, col // gs])
            Q1[:, i] = q
            W1[:, i] = dq
            err = (w - dq) / d
            if i + 1 < count:
                W1[:, i + 1:] -= err.unsqueeze(1) @ Hinv1[i, i + 1:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i1:i2] = W1
        if i2 < in_f:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return scale_all, zp_all


def main():
    ap = argparse.ArgumentParser(description="GPTQ baseline calibration (CPU)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "nac.yaml"))
    ap.add_argument("--out", required=True, help="输出前缀，如 results/gptq_qwen1.5b_g64")
    ap.add_argument("--limit-blocks", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["nac"]
    gs = args.group_size
    torch.set_num_threads(4)
    print(f"[gptq] loading {args.model} (fp32) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    samples = load_calib_samples(cfg["calib_dataset"], cfg["num_samples"], tokenizer, cfg["seqlen"])
    input_ids = torch.cat(samples, dim=0)[: cfg["capture_samples"]]
    print(f"[gptq] calib seqs={input_ids.shape[0]} seqlen={cfg['seqlen']} group={gs}", flush=True)

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
    print(f"[gptq] captured {len(block_inputs)} block inputs", flush=True)

    # 2) 逐块：块前向捕获各 Linear 输入 → H 累积 → GPTQ 量化写回 → 传播
    qparams: dict[int, dict] = {}
    t0 = time.time()
    pbar = tqdm(range(n_blocks), desc="GPTQ blocks")
    for bi in pbar:
        x = block_inputs[bi]
        hess: dict[str, torch.Tensor] = {}
        tok_counts: dict[str, int] = {}
        hooks = []

        def linear_pre_hook(name):
            def hook(_mod, aargs, _kw):
                xx = aargs[0].detach().reshape(-1, aargs[0].shape[-1]).float()
                h = hess.get(name)
                if h is None:
                    h = xx.shape[1]
                    hess[name] = torch.zeros(h, h)
                    tok_counts[name] = 0
                hess[name] += 2.0 * (xx.T @ xx)
                tok_counts[name] += xx.shape[0]
            return hook

        for name, mod in blocks[bi].named_modules():
            if isinstance(mod, LINEAR_TYPES):
                hooks.append(mod.register_forward_pre_hook(linear_pre_hook(name),
                                                           with_kwargs=True))
        with torch.no_grad():
            pos_ids = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
            pos_emb = model.model.rotary_emb(x, pos_ids)
            blocks[bi](x, position_embeddings=pos_emb)
        for h in hooks:
            h.remove()

        qp_i: dict[str, dict] = {}
        for name, mod in blocks[bi].named_modules():
            if not isinstance(mod, LINEAR_TYPES):
                continue
            H = hess[name] / tok_counts[name]  # 平均 Hessian（尺度不影响解，除法保数值稳定）
            w = mod.weight.data.float()
            scale, zp = gptq_linear(w, H, gs)
            mod.weight.data = (scale.unsqueeze(-1) * (
                torch.clamp(torch.round(
                    w.reshape(w.shape[0], -1, gs) / scale.unsqueeze(-1) + zp.unsqueeze(-1)),
                    QMIN, QMAX) - zp.unsqueeze(-1))).reshape(w.shape)
            qp_i[name] = {"scale": scale, "zp": zp}
        qparams[bi] = qp_i
        if bi + 1 < n_blocks:
            with torch.no_grad():
                block_inputs[bi + 1] = blocks[bi](
                    x, position_embeddings=pos_emb).detach().float().cpu()
        del x, hess
        pbar.set_postfix(sec=round(time.time() - t0, 0))

    # 3) 导出（与 NAC qparams 同格式）
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qparams, str(out) + ".qparams.pt")
    meta = {"model": args.model, "method": "gptq", "group_size": gs,
            "bits": 4, "asym": True, "act_order": False,
            "sequential": "block (layer-parallel within block)",
            "percdamp": PERCDAMP, "maxshrink": MAXSHRINK, "grid": GRID,
            "calib": {"dataset": cfg["calib_dataset"], "num_samples": cfg["num_samples"],
                      "capture_samples": int(input_ids.shape[0]), "seqlen": cfg["seqlen"]},
            "blocks_calibrated": n_blocks, "device": "cpu",
            "seconds": round(time.time() - t0, 1)}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[gptq] done in {meta['seconds']}s -> {out}.qparams.pt", flush=True)


if __name__ == "__main__":
    main()
