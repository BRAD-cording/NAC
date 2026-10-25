"""CPU/NPU 异构 prefill 路由器（贡献2）。

规则（方案贡献2）：输入 token 数 L：
  L <= L*   → prefill 走 NPU（量化路径）
  L >  L*   → prefill 走 CPU（llama.cpp），decode 始终走 NPU

L* 由 threshold_scan.py 实测并写入 results/route_threshold_*.json。

CPU prefill → NPU decode 的实现路径：llama.cpp prefill 得到 KV 后，
将末 token 与位置状态交给 NPU。SDK 1.1.x 的 prompt cache（rkllm_load_prompt_cache）
可承载该交接；llama.cpp 侧 KV 导出需自定义补丁。
本模块先落地"决策层 + 端到端计时"，跨后端 KV 交接在 D12–D13 实测打通。
"""
import json
import subprocess
import time
from pathlib import Path

from ..runtime.llm_client import RKLLMClient


def load_threshold(results_dir: str = "results") -> int:
    """取最新阈值扫描结果中的 L*；缺失则用保守默认 1024。"""
    cands = sorted(Path(results_dir).glob("route_threshold_*.json"),
                   key=lambda p: p.stat().st_mtime)
    for p in reversed(cands):
        data = json.loads(p.read_text())
        if data.get("L_star"):
            return int(data["L_star"])
    print("[router] 未找到阈值扫描结果，使用默认 L*=1024")
    return 1024


class HeteroRouter:
    """按 L* 决策 prefill 后端，端到端执行并计时。"""

    def __init__(self, rkllm_model: str, llama_cli: str | None = None,
                 gguf_model: str | None = None, threshold: int | None = None,
                 threads: int = 6, max_new_tokens: int = 256):
        self.threshold = threshold or load_threshold()
        self.llama_cli = llama_cli
        self.gguf = gguf_model
        self.threads = threads
        self.npu = RKLLMClient(rkllm_model, max_new_tokens=max_new_tokens)

    def route(self, n_prompt_tokens: int) -> str:
        return "npu" if n_prompt_tokens <= self.threshold else "cpu"

    def generate(self, prompt: str, n_prompt_tokens: int, max_tokens: int = 256) -> dict:
        backend = self.route(n_prompt_tokens)
        t0 = time.perf_counter()
        if backend == "npu":
            out = self.npu.generate(prompt)
            text = out["text"]
            decode_tps = out["decode_tps"]
        else:
            # CPU prefill（llama.cpp）+ NPU decode：KV 交接见模块 docstring
            proc = subprocess.run(
                [self.llama_cli, "-m", self.gguf, "-p", prompt, "-n", "1",
                 "-t", str(self.threads), "--no-display-prompt"],
                capture_output=True, text=True, timeout=3600)
            text = proc.stdout.strip()
            decode_tps = float("nan")  # 交接打通后从 NPU 回调统计
        total_ms = (time.perf_counter() - t0) * 1000
        return {"backend": backend, "text": text, "total_ms": round(total_ms, 2),
                "decode_tps": decode_tps}

    def close(self):
        self.npu.close()
