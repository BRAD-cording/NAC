"""RKLLM 运行时 ctypes 绑定（板上 RK3576 推理客户端）。

对应 rkllm-runtime-1.2.3/Linux/librkllm_api/include/rkllm.h（SDK v1.2.3，
与 HanzoHuang RK3576 w4a16 预转模型匹配）。所有评测脚本经此客户端与 NPU 交互。

结构体布局按 aarch64 LP64 对齐，与 rkllm.h v1.2.3 一致：
  RKLLMParam      = 指针 + 9×int32/float(n_keep) + mirostat3 + 2×bool + 3×char* + extend_param(120B)
  RKLLMInput      = role(ptr) + enable_thinking(bool) + input_type(int) + union(40B)
  RKLLMInferParam = mode(int) + 2×ptr + keep_history(int)
  RKLLMResult     = text* + token_id + logits(16B) + perf(20B) + last_hidden(16B)
"""
import ctypes
import time
from pathlib import Path

RKLLM_RUN_NORMAL, RKLLM_RUN_WAITING, RKLLM_RUN_FINISH, RKLLM_RUN_ERROR = 0, 1, 2, 3
RKLLM_INPUT_PROMPT, RKLLM_INPUT_TOKEN = 0, 1
RKLLM_INFER_GENERATE, RKLLM_INFER_GET_LAST_HIDDEN_LAYER, RKLLM_INFER_GET_LOGITS = 0, 1, 2

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_LIB = _REPO / "rkllm-runtime-1.2.3" / "Linux" / "librkllm_api" / "aarch64" / "librkllmrt.so"


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [("base_domain_id", ctypes.c_int32),
                ("embed_flash", ctypes.c_int8),
                ("enabled_cpus_num", ctypes.c_int8),
                ("enabled_cpus_mask", ctypes.c_uint32),
                ("n_batch", ctypes.c_uint8),
                ("use_cross_attn", ctypes.c_int8),
                ("reserved", ctypes.c_uint8 * 104)]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class _EmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]


class _TokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]


class _MultiModalInput(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("image_embed", ctypes.POINTER(ctypes.c_float)),
                ("n_image_tokens", ctypes.c_size_t)]


class _InputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p),
                ("embed_input", _EmbedInput),
                ("token_input", _TokenInput),
                ("multimodal_input", _MultiModalInput)]


class RKLLMInput(ctypes.Structure):
    _fields_ = [("role", ctypes.c_char_p),
                ("enable_thinking", ctypes.c_bool),
                ("input_type", ctypes.c_int),
                ("u", _InputUnion)]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [("mode", ctypes.c_int),
                ("lora_params", ctypes.c_void_p),
                ("prompt_cache_params", ctypes.c_void_p),
                ("keep_history", ctypes.c_int32)]


class _LastHiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)),
                ("embd_size", ctypes.c_int),
                ("num_tokens", ctypes.c_int)]


class _ResultLogits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)),
                ("vocab_size", ctypes.c_int),
                ("num_tokens", ctypes.c_int)]


class _PerfStat(ctypes.Structure):
    _fields_ = [("prefill_time_ms", ctypes.c_float),
                ("prefill_tokens", ctypes.c_int),
                ("generate_time_ms", ctypes.c_float),
                ("generate_tokens", ctypes.c_int),
                ("memory_usage_mb", ctypes.c_float)]


class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p),
                ("token_id", ctypes.c_int32),
                ("logits", _ResultLogits),
                ("perf", _PerfStat),
                ("last_hidden_layer", _LastHiddenLayer)]


CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)


def make_sink_callback(sink: dict) -> CALLBACK:
    """回调工厂：sink 累积 text / token_ids / timestamps（perf_counter 秒）。

    若 sink["capture_logits"] 为真，同时深拷贝回调中的 logits（仅 GET_LOGITS 模式
    或需要 last-token 分布时使用；generate 模式保持零拷贝以免拖慢 decode）。
    返回 0 继续推理。
    """
    sink["text"], sink["token_ids"], sink["timestamps"], sink["error"] = "", [], [], False

    @CALLBACK
    def cb(result_ptr, _userdata, state):
        now = time.perf_counter()
        if result_ptr:
            r = result_ptr.contents
            sink["timestamps"].append(now)
            if r.token_id >= 0:
                sink["token_ids"].append(int(r.token_id))
            if r.text:
                sink["text"] += r.text.decode("utf-8", "ignore")
            if sink.get("capture_logits") and r.logits.num_tokens > 0 and r.logits.logits:
                n_tok, vocab = r.logits.num_tokens, r.logits.vocab_size
                buf = ctypes.cast(r.logits.logits,
                                  ctypes.POINTER(ctypes.c_float * (n_tok * vocab))).contents
                sink["logits_out"].append({"vocab_size": vocab, "num_tokens": n_tok,
                                           "data": list(buf)})
            if state == RKLLM_RUN_FINISH:
                sink["perf"] = {"prefill_time_ms": r.perf.prefill_time_ms,
                                "prefill_tokens": r.perf.prefill_tokens,
                                "generate_time_ms": r.perf.generate_time_ms,
                                "generate_tokens": r.perf.generate_tokens,
                                "memory_usage_mb": r.perf.memory_usage_mb}
        if state == RKLLM_RUN_ERROR:
            sink["error"] = True
        return 0
    return cb


class RKLLMClient:
    """单模型实例客户端：一次 init、多次 run。with 语法自动销毁。"""

    def __init__(self, model_path: str, lib_path: str | None = None,
                 max_context_len: int = 4096, max_new_tokens: int = 512,
                 temperature: float = 0.0, top_k: int = 1, top_p: float = 0.9,
                 repeat_penalty: float = 1.0,
                 enabled_cpus_num: int = 4, enabled_cpus_mask: int = 0xF0):
        # RK3576: cpu0-3 = A53@2.0G，cpu4-7 = A72@2.2G；默认启用 4×A72 大核
        # （NPU 推理要求启用 CPU 数 >= NPU 核数(2)）
        self.lib = ctypes.CDLL(lib_path or str(DEFAULT_LIB))
        self._bind()
        self._sink: dict = {}
        self._cb = make_sink_callback(self._sink)

        # 用官方 C 函数构造默认参数（保证二进制布局一致），再按需覆盖
        p = self.lib.rkllm_createDefaultParam()
        p.model_path = str(model_path).encode()
        p.max_context_len = max_context_len
        p.max_new_tokens = max_new_tokens
        p.top_k = top_k
        p.top_p = top_p
        p.temperature = temperature if temperature > 0 else 1.0  # top_k=1 → 贪心
        p.repeat_penalty = repeat_penalty
        p.frequency_penalty = 0.0
        p.presence_penalty = 0.0
        p.mirostat = 0
        p.skip_special_token = True
        p.is_async = False
        p.extend_param.base_domain_id = 0
        p.extend_param.embed_flash = 1   # 官方 demo 默认：词嵌入走 flash
        p.extend_param.enabled_cpus_num = enabled_cpus_num
        p.extend_param.enabled_cpus_mask = enabled_cpus_mask
        p.extend_param.n_batch = 1
        p.extend_param.use_cross_attn = 0

        self.handle = ctypes.c_void_p()
        ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(p), self._cb)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed: {ret}")
        # keep_history=0：每次 run 独立（评测互不串扰）
        self.infer_param = RKLLMInferParam(mode=RKLLM_INFER_GENERATE, lora_params=None,
                                           prompt_cache_params=None, keep_history=0)

    def _bind(self):
        self.lib.rkllm_createDefaultParam.restype = RKLLMParam
        self.lib.rkllm_createDefaultParam.argtypes = []
        self.lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(RKLLMParam), CALLBACK]
        self.lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput),
                                       ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_abort.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_is_running.argtypes = [ctypes.c_void_p]

    def generate(self, prompt: str) -> dict:
        """同步生成。返回 {text, n_tokens, ttft_ms, decode_tps, total_ms}。"""
        self._sink["text"], self._sink["timestamps"] = "", []
        self._sink["token_ids"], self._sink["error"] = [], False

        inp = RKLLMInput()
        inp.role = None
        inp.enable_thinking = False
        inp.input_type = RKLLM_INPUT_PROMPT
        inp.u.prompt_input = prompt.encode("utf-8")
        t0 = time.perf_counter()
        ret = self.lib.rkllm_run(self.handle, ctypes.byref(inp),
                                 ctypes.byref(self.infer_param), None)
        total_ms = (time.perf_counter() - t0) * 1000
        if ret != 0 or self._sink["error"]:
            raise RuntimeError(f"rkllm_run failed: ret={ret} error={self._sink['error']}")

        ts = self._sink["timestamps"]
        n = len(ts)
        ttft_ms = (ts[0] - t0) * 1000 if n else float("nan")
        decode_tps = (n - 1) / (ts[-1] - ts[0]) if n > 1 else 0.0
        return {
            "text": self._sink["text"],
            "n_tokens": n,
            "token_ids": list(self._sink["token_ids"]),
            "ttft_ms": round(ttft_ms, 2),
            "decode_tps": round(decode_tps, 2),
            "total_ms": round(total_ms, 2),
        }

    def get_logits(self, prompt: str) -> dict:
        """GET_LOGITS 模式：单次前向，返回 prompt 末 token 的 logits（不生成）。

        返回 {vocab_size, num_tokens, logits(list, num_tokens*vocab), total_ms}。
        """
        self._sink["logits_out"], self._sink["error"] = [], False
        self._sink["capture_logits"] = True
        try:
            inp = RKLLMInput()
            inp.role = None
            inp.enable_thinking = False
            inp.input_type = RKLLM_INPUT_PROMPT
            inp.u.prompt_input = prompt.encode("utf-8")
            infer_param = RKLLMInferParam(mode=RKLLM_INFER_GET_LOGITS, lora_params=None,
                                          prompt_cache_params=None, keep_history=0)
            t0 = time.perf_counter()
            ret = self.lib.rkllm_run(self.handle, ctypes.byref(inp),
                                     ctypes.byref(infer_param), None)
            total_ms = (time.perf_counter() - t0) * 1000
            if ret != 0 or self._sink["error"]:
                raise RuntimeError(f"rkllm_run(GET_LOGITS) failed: ret={ret} err={self._sink['error']}")
            outs = self._sink["logits_out"]
            if not outs:
                raise RuntimeError("GET_LOGITS returned no logits")
            last = outs[-1]  # 末 token 分布
            return {"vocab_size": last["vocab_size"], "num_tokens": last["num_tokens"],
                    "logits": last["data"], "total_ms": round(total_ms, 2)}
        finally:
            self._sink["capture_logits"] = False

    def close(self):
        if getattr(self, "handle", None):
            self.lib.rkllm_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
