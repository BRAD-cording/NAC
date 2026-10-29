import sys, torch, json

for path in sys.argv[1:]:
    print("=" * 20, path)
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print("load(weights_only=True) failed:", e)
        d = torch.load(path, map_location="cpu", weights_only=False)
    print("type:", type(d))
    if isinstance(d, dict):
        ks = list(d.keys())
        print("n_keys:", len(ks))
        for k in ks[:6]:
            v = d[k]
            if isinstance(v, dict):
                print(" ", k, "{", {kk: (tuple(vv.shape) if hasattr(vv, "shape") else vv) for kk, vv in list(v.items())[:4]}, "}")
            elif hasattr(v, "shape"):
                print(" ", k, tuple(v.shape), v.dtype)
            else:
                print(" ", k, type(v), v)
        print(" ... last keys:", ks[-4:])
