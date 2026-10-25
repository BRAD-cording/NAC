"""D2 冒烟测试：加载 .rkllm → NPU 生成 → 打印 TTFT/decode 速度。

用法：python scripts/smoke_test.py --model models/xxx.rkllm [--max-new-tokens 64]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.llm_client import RKLLMClient  # noqa: E402

PROMPTS = ["你好，请用一句话介绍你自己。", "What is the capital of France?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    t0 = time.time()
    with RKLLMClient(args.model, max_new_tokens=args.max_new_tokens) as c:
        print(f"[smoke] init {time.time() - t0:.1f}s")
        for prompt in PROMPTS:
            out = c.generate(prompt)
            print(f"[smoke] {prompt!r}")
            print(f"        tokens={out['n_tokens']} ttft={out['ttft_ms']}ms "
                  f"decode={out['decode_tps']}tok/s total={out['total_ms']}ms")
            print(f"        text={out['text'][:120]!r}")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
