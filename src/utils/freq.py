"""CPU/NPU/DDR 定频控制（保证性能测量可复现，方案 D5–D6）。

封装 ../scripts/fix_freq_rk3576.sh 逻辑，也可独立使用：
  sudo python -m src.utils.freq --mode performance
"""
import argparse
import subprocess
from pathlib import Path

CPU_BASE = Path("/sys/devices/system/cpu/cpu{}/cpufreq")
NPU_DEVFREQ = Path("/sys/class/devfreq/27700000.npu/governor")
DMC_DEVFREQ = Path("/sys/class/devfreq/dmc/governor")


def _write(path: Path, value: str) -> bool:
    try:
        path.write_text(value)
        return True
    except (OSError, PermissionError) as e:
        print(f"[freq] skip {path}: {e}")
        return False


def set_performance() -> dict:
    """CPU performance governor + NPU/DMC 保持高频。需 root。"""
    applied = {}
    for i in range(8):  # RK3576: 4×A72 + 4×A53，统一 performance
        p = Path(str(CPU_BASE).format(i))
        applied[f"cpu{i}"] = _write(p / "scaling_governor", "performance")
    for name, path in (("npu", NPU_DEVFREQ), ("dmc", DMC_DEVFREQ)):
        if path.exists():
            applied[name] = _write(path, "performance")
    # NPU devfreq 路径在部分内核为 27700000.npu
    for cand in Path("/sys/class/devfreq").glob("*npu*/governor"):
        applied[f"npu:{cand.parent.name}"] = _write(cand, "performance")
    print(f"[freq] performance applied: {applied}")
    return applied


def current_freqs() -> dict:
    """读取当前频率（MHz），写入实验 JSON 用于可复现性记录。"""
    out = {}
    for i in range(8):
        f = Path(str(CPU_BASE).format(i)) / "scaling_cur_freq"
        if f.exists():
            out[f"cpu{i}_mhz"] = int(f.read_text()) // 1000
    for cand in Path("/sys/class/devfreq").glob("*npu*/cur_freq"):
        out["npu_mhz"] = int(cand.read_text()) // 1000
    dmc = Path("/sys/class/devfreq/dmc/cur_freq")
    if dmc.exists():
        out["dmc_mhz"] = int(dmc.read_text()) // 1000
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["performance", "report"], default="performance")
    args = ap.parse_args()
    if args.mode == "performance":
        set_performance()
    print(current_freqs())


if __name__ == "__main__":
    main()
