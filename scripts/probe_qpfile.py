#!/usr/bin/env python3
"""打印 qparams 文件（torch.save 格式）的真实 Schema。"""
import sys

import torch

for path in sys.argv[1:]:
    print("=" * 25, path, "=" * 25, flush=True)
    obj = torch.load(path, map_location="cpu")
    print("type:", type(obj), flush=True)
    if isinstance(obj, dict):
        print("n_keys:", len(obj), flush=True)
        for i, (k, v) in enumerate(obj.items()):
            if i < 6:
                if isinstance(v, dict):
                    print(f"  {k!r}: dict keys={list(v.keys())}", flush=True)
                    for kk, vv in v.items():
                        extra = ""
                        if hasattr(vv, "shape"):
                            extra = f" shape={tuple(vv.shape)} dtype={vv.dtype}"
                        elif isinstance(vv, (int, float, str)):
                            extra = f" value={vv}"
                        print(f"      {kk!r}: {type(vv).__name__}{extra}", flush=True)
                else:
                    extra = ""
                    if hasattr(v, "shape"):
                        extra = f" shape={tuple(v.shape)} dtype={v.dtype}"
                    elif isinstance(v, (int, float, str)):
                        extra = f" value={v}"
                    print(f"  {k!r}: {type(v).__name__}{extra}", flush=True)
        keys = list(obj.keys())
        print("first10:", keys[:10], flush=True)
        print("last5:", keys[-5:], flush=True)
    elif hasattr(obj, "keys"):
        print("n_keys:", len(obj), "first:", list(obj.keys())[:10], flush=True)
    else:
        print("repr:", repr(obj)[:500], flush=True)
