"""将 NAC 校准产物导出为 .rkllm（在装有 rkllm_toolkit 的环境执行，x86 或 cp310）。

流程（两阶段，适配板上环境：ARM 原生 python 做阶段1，qemu-x86 toolkit 做阶段2）：

阶段1（--mode gen-dequant，ARM 原生）：
  fp HF 模型 + calibrate.py 产出的 qparams.pt → 量化-反量化写回 → fp16 dequant
  HF 模型目录。语义与 validate_sim.apply_qparams(mode=nac) 完全一致：
      q = clamp(round(w/scale + zp), 0, 15);  W' = scale * (q - zp)
  每组最小/最大值必被占据（q=0 ↔ w=wmin, q=15 ↔ w=wmax），因此阶段2 toolkit
  normal(min-max) 重量化可精确还原 NAC 的 scale/zp（fp16 存储舍入 ~1e-3 相对
  误差远低于 4bit 量化步长，可忽略）。

阶段2（x86 toolkit，见 scripts/rkllm_x86/build_rkllm.py）：
  RKLLM().load_huggingface(dequant目录) → build(do_quantization=True,
  quantized_dtype=w4a16_g64, quantized_algorithm=normal) → export_rkllm()

用法（阶段1）：
  python -m src.nac.export_rkllm --mode gen-dequant \
      --model models/Qwen2.5-1.5B-Instruct-hf \
      --qparams results/nac_qwen1.5b_g64.qparams.pt \
      --group-size 64 --out models/nac_dequant_g64
"""
import argparse
import json
from pathlib import Path

import torch


def verify_toolkit() -> None:
    try:
        from rkllm.api import RKLLM  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "未找到 rkllm_toolkit。请在 x86 环境 pip install "
            "rkllm-toolkit/packages/rkllm_toolkit-1.1.4-cp310-cp310-linux_x86_64.whl"
        ) from e


def load_model_with_nac_weights(model_id: str, weights_pt: str):
    """把 NAC 校准后的 dequant 权重载入 HF 模型（路径 A）。"""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    state = torch.load(weights_pt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[export] unexpected keys: {unexpected[:5]} ...")
    if missing:
        print(f"[export] missing keys: {missing[:5]} ...")
    return model


def save_dequant_model(model, model_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    # tokenizer 一并保存，toolkit 需要
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[export] tokenizer save skipped: {e}")
    return out_dir


def gen_dequant_from_qparams(model_id: str, qparams_pt: str, group_size: int,
                              out_dir: Path) -> Path:
    """阶段1：fp 权重 + NAC qparams → 量化-反量化写回 → fp16 dequant HF 目录。"""
    from src.nac.quantize import dequantize_weight, quantize_weight
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qparams = torch.load(qparams_pt, map_location="cpu", weights_only=True)
    print(f"[gen-dequant] loading {model_id} (fp32) ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    n = 0
    with torch.no_grad():
        for bi, block in enumerate(model.model.layers):
            for name, mod in block.named_modules():
                if not isinstance(mod, torch.nn.Linear):
                    continue
                qp = qparams[bi][name]
                res = quantize_weight(mod.weight.data.float(), group_size,
                                      qp["scale"], qp["zp"])
                mod.weight.data = dequantize_weight(res.q, res.scale, res.zp)
                n += 1
            print(f"[gen-dequant] block {bi} done", flush=True)
    print(f"[gen-dequant] {n} linears dequantized (gs={group_size})")

    model = model.half()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    # toolkit 1.2.3 只认 transformers 4.x 的 config/tokenizer 格式（5.x 的
    # layer_types 列表会让其崩溃），统一用源模型文件覆盖，仅 torch_dtype 改 fp16
    import json
    import shutil
    src = Path(model_id)
    for fn in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt"):
        if (src / fn).exists():
            shutil.copy(src / fn, out_dir / fn)
    cfg = json.loads((src / "config.json").read_text())
    cfg["torch_dtype"] = "float16"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    for junk in ("chat_template.jinja",):
        (out_dir / junk).unlink(missing_ok=True)
    print("[gen-dequant] config/tokenizer replaced with source (transformers 4.x format)")

    # 体检：抽样组反量值应严格落在 NAC 网格上，且角点占据（min=-scale*zp, max=scale*(15-zp)）
    w = dict(model.named_parameters())["model.layers.0.self_attn.q_proj.weight"].data
    qp = qparams[0]["self_attn.q_proj"]
    g = w.float().reshape(w.shape[0], -1, group_size)
    lo, hi = -qp["scale"] * qp["zp"], qp["scale"] * (15 - qp["zp"])
    print(f"[gen-dequant] check blk0.q_proj: min_err={float((g.amin(-1) - lo).abs().max()):.3e} "
          f"max_err={float((g.amax(-1) - hi).abs().max()):.3e}")
    print(f"[gen-dequant] OK -> {out_dir}")
    return out_dir


def build_rkllm(model_path: Path, dataset: str, cfg: dict, out: Path,
                extra_qparams: str | None = None) -> None:
    from rkllm.api import RKLLM
    llm = RKLLM()
    ret = llm.load_huggingface(model=str(model_path), model_lora=None, device="cpu")
    if ret != 0:
        raise SystemExit(f"load_huggingface failed: {ret}")
    kwargs = dict(
        do_quantization=cfg["do_quantization"],
        optimization_level=cfg["optimization_level"],
        quantized_dtype=cfg["quantized_dtype"],
        quantized_algorithm=cfg["quantized_algorithm"],
        target_platform=cfg["target_platform"],
        dataset=dataset,
    )
    if extra_qparams:
        kwargs["extra_qparams"] = extra_qparams
    ret = llm.build(**kwargs)
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")
    ret = llm.export_rkllm(str(out))
    if ret != 0:
        raise SystemExit(f"export_rkllm failed: {ret}")
    print(f"[export] OK -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Export NAC-calibrated model to .rkllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--nac", default=None, help="calibrate.py 产物 *.weights.pt（旧路径）")
    ap.add_argument("--qparams", default=None, help="calibrate.py 产物 *.qparams.pt（gen-dequant 用）")
    ap.add_argument("--group-size", type=int, default=64, help="NAC 校准 group size")
    ap.add_argument("--dataset", required=False, help="校准样本 JSON（toolkit build 用）")
    ap.add_argument("--out", required=True, help="输出 .rkllm 路径")
    ap.add_argument("--mode", choices=["dequant", "qparams", "gen-dequant"], default="dequant")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "nac.yaml"))
    ap.add_argument("--workdir", default="models/_dequant_export")
    args = ap.parse_args()

    if args.mode == "gen-dequant":
        if not args.qparams:
            raise SystemExit("--mode gen-dequant 需要 --qparams")
        gen_dequant_from_qparams(args.model, args.qparams, args.group_size,
                                 Path(args.out))
        print(f"[gen-dequant] 阶段1完成。阶段2（x86 toolkit）:\n"
              f"  build(do_quantization=True, quantized_dtype=w4a16_g{args.group_size},\n"
              f"        quantized_algorithm=normal, target_platform=rk3576) -> {args.out}")
        return

    verify_toolkit()
    import yaml
    with open(args.config) as f:
        ecfg = yaml.safe_load(f)["export"]

    if args.mode == "dequant":
        model = load_model_with_nac_weights(args.model, args.nac)
        workdir = Path(args.workdir)
        print(f"[export] saving dequant model -> {workdir}")
        model_path = save_dequant_model(model, args.model, workdir)
        # dequant 后权重已是目标精度，toolkit 只做格式转换
        build_rkllm(model_path, args.dataset,
                    {**ecfg, "do_quantization": False}, Path(args.out))
    else:
        # 路径 B：NAC scale/zp → gdq.qparams（格式需对照 toolkit 源码核实）
        qpath = Path(args.out).with_suffix(".qparams")
        meta = json.loads(Path(str(args.nac).replace(".weights.pt", ".meta.json")).read_text())
        print(f"[export] qparams mode: 请核实 gdq.qparams 格式后生成 {qpath} "
              f"(group={meta['group_size']})；本脚本暂以 toolkit 默认量化 + NAC qparams 占位")
        build_rkllm(Path(args.model), args.dataset, ecfg, Path(args.out),
                    extra_qparams=str(qpath) if qpath.exists() else None)


if __name__ == "__main__":
    main()
