#!/usr/bin/env python3
"""列出 qparams 文件的全部键（名称/类型/形状），用于确认 zeropoint 的键名。"""
import sys

import torch

obj = torch.load(sys.argv[1], map_location="cpu")
print("n_keys:", len(obj), flush=True)
for k, v in obj.items():
    if hasattr(v, "shape"):
        print(f"{k}: {type(v).__name__} {tuple(v.shape)} {v.dtype}", flush=True)
    else:
        print(f"{k}: {type(v).__name__} {v}", flush=True)
