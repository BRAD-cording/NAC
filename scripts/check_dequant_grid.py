"""轻量验证 dequant 模型是否严格落在 NAC 网格上（不重载整模型）。

用法：python -m scripts.check_dequant_grid --safetensors models/nac_dequant_g64/model.safetensors \
        --qparams results/nac_qwen1.5b_g64.qparams.pt --group-size 64
"""
import argparse

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--qparams", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--samples", default=[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.13.mlp.down_proj.weight",
        "model.layers.27.mlp.gate_proj.weight",
    ])
    args = ap.parse_args()

    qparams = torch.load(args.qparams, map_location="cpu", weights_only=True)
    gs = args.group_size
    ok = True
    with safe_open(args.safetensors, framework="pt", device="cpu") as f:
        for key in args.samples:
            bi = int(key.split(".")[2])
            name = ".".join(key.split(".")[3:-1])
            qp = qparams[bi][name]
            w = f.get_tensor(key).float()
            g = w.reshape(w.shape[0], -1, gs)
            lo, hi = -qp["scale"] * qp["zp"], qp["scale"] * (15 - qp["zp"])
            err_lo = float((g.amin(-1) - lo).abs().max())
            err_hi = float((g.amax(-1) - hi).abs().max())
            # 网格成员：抽样行应满足 w == scale*(round(w/scale+zp)-zp)（fp16 舍入级误差）
            sub = g[::max(1, g.shape[0] // 8)]
            st = max(1, qp["scale"].shape[0] // 8)
            sc, zp = qp["scale"][::st], qp["zp"][::st]
            q_rec = torch.round(sub / sc.unsqueeze(-1) + zp.unsqueeze(-1))
            grid_err = float((sc.unsqueeze(-1) * (q_rec - zp.unsqueeze(-1)) - sub).abs().max())
            status = "OK" if err_lo < 1e-2 and err_hi < 1e-2 and grid_err < 1e-2 else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"[{status}] {key}: min_err={err_lo:.3e} max_err={err_hi:.3e} "
                  f"grid_err={grid_err:.3e} scale_mean={qp['scale'].mean():.3e}")
    print("GRID-CHECK:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
