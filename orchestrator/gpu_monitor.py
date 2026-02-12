"""
GPU Monitor Module
==================
Real-time monitoring of GPU metrics via pynvml (nvidia-ml-py).
Tracks memory usage, utilization, temperature, power, and clock speeds.
Provides historical snapshots for the rule-based controller.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUSnapshot:
    """Single-point-in-time GPU metrics."""

    gpu_id: int
    timestamp: float
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_pct: float
    utilization_pct: float
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    clock_sm_mhz: int
    clock_mem_mhz: int


@dataclass
class GPUHistory:
    """Rolling window of GPU snapshots for trend analysis."""

    max_snapshots: int = 120  # ~20 minutes at 10s intervals
    snapshots: deque = field(default_factory=lambda: deque(maxlen=120))

    def add(self, snapshot: GPUSnapshot):
        self.snapshots.append(snapshot)

    @property
    def peak_memory_pct(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(s.memory_pct for s in self.snapshots)

    @property
    def avg_memory_pct(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s.memory_pct for s in self.snapshots) / len(self.snapshots)

    @property
    def avg_utilization(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s.utilization_pct for s in self.snapshots) / len(self.snapshots)

    @property
    def latest(self) -> Optional[GPUSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def memory_trend(self) -> float:
        """Returns memory change rate (% per minute). Positive = increasing."""
        if len(self.snapshots) < 10:
            return 0.0
        recent = list(self.snapshots)[-10:]
        dt = recent[-1].timestamp - recent[0].timestamp
        if dt < 1.0:
            return 0.0
        dm = recent[-1].memory_pct - recent[0].memory_pct
        return (dm / dt) * 60.0  # % per minute


class GPUMonitor:
    """
    Monitors all visible GPUs using pynvml.
    Maintains per-GPU rolling history for trend analysis.
    Thread-safe for use with the orchestrator's polling loop.
    """

    def __init__(self, history_size: int = 120):
        self._initialized = False
        self._gpu_count = 0
        self._handles = []
        self._histories: Dict[int, GPUHistory] = {}
        self._lock = threading.Lock()
        self._history_size = history_size
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._init_nvml()

    def _init_nvml(self):
        if not PYNVML_AVAILABLE:
            logger.warning("pynvml not available; GPU monitoring will use mock data")
            return

        try:
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self._gpu_count)
            ]
            for i in range(self._gpu_count):
                self._histories[i] = GPUHistory(max_snapshots=self._history_size)
            self._initialized = True
            logger.info(f"GPU monitor initialized: {self._gpu_count} GPU(s) detected")
        except pynvml.NVMLError as e:
            logger.error(f"NVML init failed: {e}")
            self._initialized = False

    def poll(self) -> Dict[int, GPUSnapshot]:
        """Poll all GPUs once and return current snapshots."""
        snapshots = {}
        ts = time.time()

        if not self._initialized:
            return self._mock_snapshots(ts)

        with self._lock:
            for i, handle in enumerate(self._handles):
                try:
                    snap = self._read_gpu(i, handle, ts)
                    snapshots[i] = snap
                    self._histories[i].add(snap)
                except pynvml.NVMLError as e:
                    logger.warning(f"Failed to read GPU {i}: {e}")

        return snapshots

    def _read_gpu(self, idx: int, handle, ts: float) -> GPUSnapshot:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        except pynvml.NVMLError:
            power = 0.0

        try:
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except pynvml.NVMLError:
            power_limit = 0.0

        try:
            clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except pynvml.NVMLError:
            clock_sm = clock_mem = 0

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem_used_mb = mem_info.used / (1024 ** 2)
        mem_total_mb = mem_info.total / (1024 ** 2)
        mem_pct = (mem_info.used / mem_info.total) * 100.0

        return GPUSnapshot(
            gpu_id=idx,
            timestamp=ts,
            name=name,
            memory_used_mb=mem_used_mb,
            memory_total_mb=mem_total_mb,
            memory_pct=mem_pct,
            utilization_pct=float(util.gpu),
            temperature_c=temp,
            power_draw_w=power,
            power_limit_w=power_limit,
            clock_sm_mhz=clock_sm,
            clock_mem_mhz=clock_mem,
        )

    def _mock_snapshots(self, ts: float) -> Dict[int, GPUSnapshot]:
        """Return mock data when NVML is unavailable (for testing)."""
        import random

        mock = {}
        for i in range(max(1, self._gpu_count)):
            snap = GPUSnapshot(
                gpu_id=i,
                timestamp=ts,
                name="Mock-GPU",
                memory_used_mb=random.uniform(2000, 7000),
                memory_total_mb=8192.0,
                memory_pct=random.uniform(40, 90),
                utilization_pct=random.uniform(20, 95),
                temperature_c=random.randint(40, 80),
                power_draw_w=random.uniform(50, 200),
                power_limit_w=250.0,
                clock_sm_mhz=random.randint(1000, 1800),
                clock_mem_mhz=random.randint(800, 1200),
            )
            mock[i] = snap
            if i not in self._histories:
                self._histories[i] = GPUHistory(max_snapshots=self._history_size)
            self._histories[i].add(snap)
        return mock

    def get_history(self, gpu_id: int) -> Optional[GPUHistory]:
        return self._histories.get(gpu_id)

    def get_all_histories(self) -> Dict[int, GPUHistory]:
        return dict(self._histories)

    @property
    def gpu_count(self) -> int:
        return max(1, self._gpu_count)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def start_background(self, interval_sec: float = 10.0):
        """Start background polling thread."""
        if self._bg_thread and self._bg_thread.is_alive():
            return

        self._stop_event.clear()

        def _poll_loop():
            while not self._stop_event.is_set():
                try:
                    self.poll()
                except Exception as e:
                    logger.error(f"GPU poll error: {e}")
                self._stop_event.wait(interval_sec)

        self._bg_thread = threading.Thread(target=_poll_loop, daemon=True)
        self._bg_thread.start()
        logger.info(f"GPU monitor background thread started (interval={interval_sec}s)")

    def stop_background(self):
        """Stop background polling thread."""
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=5)
            logger.info("GPU monitor background thread stopped")

    def summary(self) -> str:
        """Human-readable summary of current GPU state."""
        snapshots = self.poll()
        lines = ["=" * 60, "GPU Monitor Summary", "=" * 60]
        for gpu_id, snap in sorted(snapshots.items()):
            hist = self._histories.get(gpu_id)
            lines.append(
                f"  GPU {gpu_id} ({snap.name}):\n"
                f"    Memory: {snap.memory_used_mb:.0f}/{snap.memory_total_mb:.0f} MB "
                f"({snap.memory_pct:.1f}%)\n"
                f"    Utilization: {snap.utilization_pct:.0f}%\n"
                f"    Temperature: {snap.temperature_c}°C\n"
                f"    Power: {snap.power_draw_w:.0f}/{snap.power_limit_w:.0f} W"
            )
            if hist and len(hist.snapshots) > 1:
                lines.append(
                    f"    Mem trend: {hist.memory_trend:+.2f} %/min | "
                    f"Peak mem: {hist.peak_memory_pct:.1f}%"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    def shutdown(self):
        self.stop_background()
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
