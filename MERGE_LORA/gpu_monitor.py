import pynvml
import time
import threading
from typing import Optional, Tuple

class GPUMonitor:
    """Мониторинг мощности и загрузки GPU через NVML."""
    def __init__(self, device_id: int = 0, sampling_interval: float = 0.1):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.interval = sampling_interval
        self._stop_event = None
        self._thread = None
        self.power_readings = []
        self.util_readings = []

    def _sample(self):
        while not self._stop_event.is_set():
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # мВт -> Вт
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                self.power_readings.append(power)
                self.util_readings.append(util)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> Tuple[float, float, float]:
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        pynvml.nvmlShutdown()
        avg_power = sum(self.power_readings) / len(self.power_readings) if self.power_readings else 0.0
        max_power = max(self.power_readings) if self.power_readings else 0.0
        avg_util = sum(self.util_readings) / len(self.util_readings) if self.util_readings else 0.0
        return avg_power, max_power, avg_util