"""
Log Parser Module
=================
Parses training logs in real-time for:
  - CUDA OOM errors
  - NCCL communication errors / timeouts
  - Loss values and training metrics
  - DeepSpeed-specific messages
  - Gradient overflow / NaN detection
"""

import re
import os
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum, auto
from collections import deque

logger = logging.getLogger(__name__)


class EventType(Enum):
    OOM = auto()
    NCCL_ERROR = auto()
    NCCL_TIMEOUT = auto()
    NAN_LOSS = auto()
    GRAD_OVERFLOW = auto()
    CUDA_ERROR = auto()
    DEEPSPEED_ERROR = auto()
    CHECKPOINT_SAVED = auto()
    TRAINING_STEP = auto()
    EVAL_RESULT = auto()
    INFO = auto()
    WARNING = auto()


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class LogEvent:
    """Parsed event from a training log line."""

    event_type: EventType
    severity: Severity
    message: str
    timestamp: float
    raw_line: str
    metadata: dict = field(default_factory=dict)


# --- Compiled regex patterns for fast matching ---

PATTERNS = {
    EventType.OOM: [
        re.compile(r"CUDA out of memory", re.IGNORECASE),
        re.compile(r"RuntimeError:.*out of memory", re.IGNORECASE),
        re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
        re.compile(r"OOM", re.IGNORECASE),
        re.compile(r"Tried to allocate.*GiB", re.IGNORECASE),
    ],
    EventType.NCCL_ERROR: [
        re.compile(r"NCCL\s+error", re.IGNORECASE),
        re.compile(r"ncclInternalError", re.IGNORECASE),
        re.compile(r"ncclSystemError", re.IGNORECASE),
        re.compile(r"NCCL.*unhandled", re.IGNORECASE),
        re.compile(r"ProcessGroupNCCL.*error", re.IGNORECASE),
    ],
    EventType.NCCL_TIMEOUT: [
        re.compile(r"NCCL.*timeout", re.IGNORECASE),
        re.compile(r"Watchdog caught.*NCCL", re.IGNORECASE),
        re.compile(r"ProcessGroupNCCL.*timed out", re.IGNORECASE),
        re.compile(r"NCCL_ASYNC_ERROR_HANDLING", re.IGNORECASE),
    ],
    EventType.NAN_LOSS: [
        re.compile(r"loss.*nan", re.IGNORECASE),
        re.compile(r"nan.*loss", re.IGNORECASE),
        re.compile(r"NaN detected", re.IGNORECASE),
    ],
    EventType.GRAD_OVERFLOW: [
        re.compile(r"[Gg]radient overflow"),
        re.compile(r"Grad overflow", re.IGNORECASE),
        re.compile(r"overflow_check", re.IGNORECASE),
        re.compile(r"skipped.*overflow", re.IGNORECASE),
    ],
    EventType.CUDA_ERROR: [
        re.compile(r"CUDA error", re.IGNORECASE),
        re.compile(r"cudaError", re.IGNORECASE),
        re.compile(r"device-side assert", re.IGNORECASE),
    ],
    EventType.DEEPSPEED_ERROR: [
        re.compile(r"deepspeed.*error", re.IGNORECASE),
        re.compile(r"DeepSpeed.*failed", re.IGNORECASE),
    ],
    EventType.CHECKPOINT_SAVED: [
        re.compile(r"[Ss]aved checkpoint"),
        re.compile(r"[Mm]odel saved"),
        re.compile(r"Saving model checkpoint", re.IGNORECASE),
    ],
}

# Pattern to extract training step metrics
STEP_PATTERN = re.compile(
    r"step[\s=:]+(\d+).*?loss[\s=:]+([0-9.eE+-]+)", re.IGNORECASE
)
THROUGHPUT_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(?:samples?|tokens?|examples?)/s(?:ec)?", re.IGNORECASE
)
LR_PATTERN = re.compile(r"lr[\s=:]+([0-9.eE+-]+)", re.IGNORECASE)

# Severity mapping
SEVERITY_MAP = {
    EventType.OOM: Severity.FATAL,
    EventType.NCCL_ERROR: Severity.FATAL,
    EventType.NCCL_TIMEOUT: Severity.CRITICAL,
    EventType.NAN_LOSS: Severity.CRITICAL,
    EventType.GRAD_OVERFLOW: Severity.WARNING,
    EventType.CUDA_ERROR: Severity.FATAL,
    EventType.DEEPSPEED_ERROR: Severity.CRITICAL,
    EventType.CHECKPOINT_SAVED: Severity.INFO,
    EventType.TRAINING_STEP: Severity.INFO,
    EventType.EVAL_RESULT: Severity.INFO,
    EventType.INFO: Severity.INFO,
    EventType.WARNING: Severity.WARNING,
}


class LogParser:
    """
    Real-time training log parser with callback support.

    Tails a log file and emits structured events when patterns match.
    Thread-safe. Supports multiple event callbacks.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._events: deque = deque(maxlen=10000)
        self._callbacks: List[Callable[[LogEvent], None]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tail_thread: Optional[threading.Thread] = None
        self._file_pos = 0

        # Aggregated metrics
        self._step_losses: List[tuple] = []  # (step, loss)
        self._oom_count = 0
        self._nccl_error_count = 0
        self._grad_overflow_count = 0

    def register_callback(self, callback: Callable[[LogEvent], None]):
        """Register a callback invoked for every parsed event."""
        self._callbacks.append(callback)

    def parse_line(self, line: str) -> Optional[LogEvent]:
        """Parse a single log line and return an event if a pattern matches."""
        line = line.strip()
        if not line:
            return None

        ts = time.time()

        # Check error/event patterns (highest severity first)
        for event_type in [
            EventType.OOM,
            EventType.NCCL_ERROR,
            EventType.NCCL_TIMEOUT,
            EventType.CUDA_ERROR,
            EventType.DEEPSPEED_ERROR,
            EventType.NAN_LOSS,
            EventType.GRAD_OVERFLOW,
            EventType.CHECKPOINT_SAVED,
        ]:
            for pattern in PATTERNS[event_type]:
                if pattern.search(line):
                    event = LogEvent(
                        event_type=event_type,
                        severity=SEVERITY_MAP[event_type],
                        message=line,
                        timestamp=ts,
                        raw_line=line,
                    )
                    self._record_event(event)
                    return event

        # Check for training step metrics
        step_match = STEP_PATTERN.search(line)
        if step_match:
            step = int(step_match.group(1))
            loss = float(step_match.group(2))
            meta = {"step": step, "loss": loss}

            # Extract throughput if present
            tp_match = THROUGHPUT_PATTERN.search(line)
            if tp_match:
                meta["throughput"] = float(tp_match.group(1))

            # Extract learning rate if present
            lr_match = LR_PATTERN.search(line)
            if lr_match:
                meta["lr"] = float(lr_match.group(1))

            event = LogEvent(
                event_type=EventType.TRAINING_STEP,
                severity=Severity.INFO,
                message=f"Step {step}: loss={loss:.4f}",
                timestamp=ts,
                raw_line=line,
                metadata=meta,
            )
            with self._lock:
                self._step_losses.append((step, loss))
            self._record_event(event)
            return event

        return None

    def _record_event(self, event: LogEvent):
        """Record event and invoke callbacks."""
        with self._lock:
            self._events.append(event)

            if event.event_type == EventType.OOM:
                self._oom_count += 1
            elif event.event_type in (EventType.NCCL_ERROR, EventType.NCCL_TIMEOUT):
                self._nccl_error_count += 1
            elif event.event_type == EventType.GRAD_OVERFLOW:
                self._grad_overflow_count += 1

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def parse_file(self, path: Optional[str] = None) -> List[LogEvent]:
        """Parse an entire log file and return all events."""
        path = path or self.log_path
        if not path or not os.path.exists(path):
            return []

        events = []
        with open(path, "r") as f:
            for line in f:
                event = self.parse_line(line)
                if event:
                    events.append(event)
        return events

    def start_tailing(self, path: Optional[str] = None, poll_interval: float = 1.0):
        """Start tailing a log file in a background thread."""
        path = path or self.log_path
        if not path:
            logger.error("No log path specified for tailing")
            return

        self.log_path = path
        self._stop_event.clear()

        def _tail_loop():
            logger.info(f"Log parser tailing: {path}")
            while not self._stop_event.is_set():
                try:
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            f.seek(self._file_pos)
                            for line in f:
                                self.parse_line(line)
                            self._file_pos = f.tell()
                except Exception as e:
                    logger.error(f"Log tail error: {e}")
                self._stop_event.wait(poll_interval)

        self._tail_thread = threading.Thread(target=_tail_loop, daemon=True)
        self._tail_thread.start()

    def stop_tailing(self):
        """Stop background tailing."""
        self._stop_event.set()
        if self._tail_thread:
            self._tail_thread.join(timeout=5)

    @property
    def oom_count(self) -> int:
        return self._oom_count

    @property
    def nccl_error_count(self) -> int:
        return self._nccl_error_count

    @property
    def grad_overflow_count(self) -> int:
        return self._grad_overflow_count

    @property
    def recent_events(self) -> List[LogEvent]:
        """Return the last 50 events."""
        with self._lock:
            return list(self._events)[-50:]

    @property
    def fatal_events(self) -> List[LogEvent]:
        """Return all fatal-severity events."""
        with self._lock:
            return [e for e in self._events if e.severity == Severity.FATAL]

    @property
    def latest_loss(self) -> Optional[float]:
        with self._lock:
            return self._step_losses[-1][1] if self._step_losses else None

    @property
    def latest_step(self) -> Optional[int]:
        with self._lock:
            return self._step_losses[-1][0] if self._step_losses else None

    def loss_history(self, last_n: int = 100) -> List[tuple]:
        """Return recent (step, loss) pairs."""
        with self._lock:
            return list(self._step_losses[-last_n:])

    def is_loss_diverging(self, window: int = 20, threshold: float = 1.5) -> bool:
        """Check if loss is diverging (increasing significantly over recent window)."""
        with self._lock:
            if len(self._step_losses) < window:
                return False
            recent = self._step_losses[-window:]
            first_half = [l for _, l in recent[: window // 2]]
            second_half = [l for _, l in recent[window // 2 :]]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            return avg_second > avg_first * threshold

    def summary(self) -> str:
        """Human-readable summary of parsed events."""
        lines = [
            "=" * 60,
            "Log Parser Summary",
            "=" * 60,
            f"  OOM events:         {self._oom_count}",
            f"  NCCL errors:        {self._nccl_error_count}",
            f"  Grad overflows:     {self._grad_overflow_count}",
            f"  Total events:       {len(self._events)}",
        ]
        if self._step_losses:
            step, loss = self._step_losses[-1]
            lines.append(f"  Latest step/loss:   {step} / {loss:.4f}")
            lines.append(
                f"  Loss diverging:     {self.is_loss_diverging()}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self):
        """Reset all counters and events."""
        with self._lock:
            self._events.clear()
            self._step_losses.clear()
            self._oom_count = 0
            self._nccl_error_count = 0
            self._grad_overflow_count = 0
            self._file_pos = 0
