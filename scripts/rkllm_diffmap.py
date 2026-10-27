#!/usr/bin/env python3
"""rkllm 字节 diff 地图：64KiB 块粒度扫描两模型，输出差异区块分布。
用法: python scripts/rkllm_diffmap.py A.rkllm B.rkllm [block_kib]
"""
import sys

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    blk = int(sys.argv[3]) * 1024 if len(sys.argv) > 3 else 64 * 1024
    fa = open(a_path, "rb")
    fb = open(b_path, "rb")
    off = 0
    diffs = []  # (start, end) 连续差异区
    cur = None
    while True:
        a = fa.read(blk)
        b = fb.read(blk)
        if not a and not b:
            break
        if a != b:
            if cur is None:
                cur = off
            cur_end = off + max(len(a), len(b))
        else:
            if cur is not None:
                diffs.append((cur, cur_end))
                cur = None
        off += blk
    if cur is not None:
        diffs.append((cur, cur_end))
    sa = fa.seek(0, 2); sz_a = fa.tell()
    sz_b = fb.tell()
    print(f"A={a_path} {sz_a} B")
    print(f"B={b_path} {sz_b} B")
    print(f"block={blk//1024}KiB diff_regions={len(diffs)}")
    total = sum(e - s for s, e in diffs)
    print(f"diff_total={total} B ({total*100/max(sz_a,sz_b):.1f}%)")
    for s, e in diffs:
        print(f"  0x{s:09x}-0x{e:09x}  {s/1048576:8.1f}-{e/1048576:8.1f} MiB  ({(e-s)//1024} KiB)")

if __name__ == "__main__":
    main()
