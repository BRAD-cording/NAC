"""功耗采样（方案 3.3：功耗、能效 tokens/J）。

板上实测结论（NanoPi M5 / RK3576，2026-10）：
- /sys/class/power_supply/simple-vin 仅提供电压（in0_input ≈ 4.9V），无电流通道
- 无 /sys/class/powercap；hwmon 无 power*_input → 板载无法直接测整机功率
因此功耗数据需外接 USB 功率计（INA260 / UM34C 等），本类提供两种路径：
1. hwmon/power_supply 节点探测（外接 INA 驱动后自动可用）
2. 电压采样兜底（论文中可作供电稳定性佐证，不作为功耗结论）

能效 tokens/J 若无功率计，论文改为报告 tokens/s/W 或标注 limitation。
"""
import threading
import time
from pathlib import Path


def _find_power_nodes() -> list[Path]:
    """功率节点（µW/W 可推算）；若无则回退电压节点。"""
    nodes = []
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        nodes += sorted(hw.glob("power*_input"))
    for ps in Path("/sys/class/power_supply").glob("*"):
        nodes += sorted(ps.glob("power_now"))
    return nodes


def _find_voltage_nodes() -> list[Path]:
    nodes = []
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        nodes += sorted(hw.glob("in*_input"))
    for ps in Path("/sys/class/power_supply").glob("*"):
        nodes += sorted(ps.glob("voltage_now"))
    return nodes


def _read_uw(path: Path) -> float | None:
    try:
        v = int(path.read_text().strip())
        return v / 1e6  # µW → W
    except (OSError, ValueError):
        return None


def _read_v(path: Path) -> float | None:
    try:
        v = int(path.read_text().strip())
        return v / 1000.0  # hwmon in*_input 为 mV；power_supply voltage_now 为 µV
    except (OSError, ValueError):
        return None


class PowerSampler:
    """后台线程周期采样所有功耗节点，stop() 返回均值功耗与累计能量。"""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.nodes = _find_power_nodes()
        self.voltage_nodes = _find_voltage_nodes()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[float] = []
        self.voltages: list[float] = []
        if not self.nodes:
            print("[power] 板载无功率节点（实测：simple-vin 仅电压）。"
                  "将仅记录电压；整机功耗请外接 INA260/UM34C 后重测")

    def _loop(self):
        while not self._stop.is_set():
            total = 0.0
            for n in self.nodes:
                w = _read_uw(n)
                if w is not None:
                    total += w
            if total > 0:
                self.samples.append(total)
            if self.voltage_nodes:
                v = _read_v(self.voltage_nodes[0])
                if v is not None:
                    self.voltages.append(v)
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self.voltages.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, results_tokens: int = 0) -> tuple[float | None, float | None]:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2)
        self.last_voltage_v = (round(sum(self.voltages) / len(self.voltages), 3)
                               if self.voltages else None)
        if not self.samples:
            return None, None
        mean_w = sum(self.samples) / len(self.samples)
        energy_j = mean_w * (len(self.samples) * self.interval) if self.interval else None
        return round(mean_w, 3), round(energy_j, 3)


def sample_static_power(duration: float = 5.0) -> float | None:
    """静态功耗（idle）采样，用于能效基线扣除。"""
    s = PowerSampler(interval=0.2)
    if not s.nodes:
        return None
    s.start()
    time.sleep(duration)
    w, _ = s.stop()
    return w
