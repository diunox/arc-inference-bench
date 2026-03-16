"""GPU monitoring helpers for Intel Arc A770 via sysfs hwmon."""

import time

HWMON = "/sys/class/drm/card1/device/hwmon"
_hwmon_path = None


def _find_hwmon():
    global _hwmon_path
    if _hwmon_path:
        return _hwmon_path
    import os
    for d in os.listdir(HWMON):
        p = os.path.join(HWMON, d)
        if os.path.isfile(os.path.join(p, "energy1_input")):
            _hwmon_path = p
            return p
    raise FileNotFoundError("No hwmon with energy1_input found")


def read_energy_uj() -> int:
    """Read cumulative GPU energy in microjoules."""
    with open(f"{_find_hwmon()}/energy1_input") as f:
        return int(f.read().strip())


def read_gpu_temp() -> float:
    """Read GPU temperature in Celsius (millidegrees → degrees)."""
    with open(f"{_find_hwmon()}/temp1_input") as f:
        return int(f.read().strip()) / 1000.0


def read_fan_rpm() -> int:
    """Read GPU fan speed in RPM."""
    with open(f"{_find_hwmon()}/fan1_input") as f:
        return int(f.read().strip())


class PowerSample:
    """Context manager that measures energy, time, and temperature over a block."""

    def __init__(self):
        self.energy_start = 0
        self.energy_end = 0
        self.time_start = 0.0
        self.time_end = 0.0
        self.temp_start = 0.0
        self.temp_end = 0.0

    def __enter__(self):
        self.energy_start = read_energy_uj()
        self.temp_start = read_gpu_temp()
        self.time_start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.time_end = time.monotonic()
        self.energy_end = read_energy_uj()
        self.temp_end = read_gpu_temp()

    @property
    def elapsed_s(self) -> float:
        return self.time_end - self.time_start

    @property
    def energy_j(self) -> float:
        delta = self.energy_end - self.energy_start
        if delta < 0:  # counter wraparound
            delta += 2**64
        return delta / 1_000_000.0

    @property
    def avg_watts(self) -> float:
        return self.energy_j / self.elapsed_s if self.elapsed_s > 0 else 0

    @property
    def temp_delta(self) -> float:
        return self.temp_end - self.temp_start

    def to_dict(self) -> dict:
        return {
            "energy_j": round(self.energy_j, 2),
            "elapsed_s": round(self.elapsed_s, 2),
            "avg_watts": round(self.avg_watts, 1),
            "temp_start_c": round(self.temp_start, 1),
            "temp_end_c": round(self.temp_end, 1),
            "temp_delta_c": round(self.temp_delta, 1),
        }
