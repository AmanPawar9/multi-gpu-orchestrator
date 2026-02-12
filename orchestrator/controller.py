"""
Rule-Based Controller (Agentic Controller)
===========================================
Autonomous decision engine that monitors GPU metrics and training logs,
then adapts job parameters to keep training stable and efficient.

Decision rules:
  1. OOM handling:     reduce batch_size → enable activation checkpointing → increase ZeRO stage
  2. Memory pressure:  proactively reduce batch_size before OOM
  3. Underutilization: increase batch_size to improve GPU efficiency
  4. NCCL errors:      requeue the job with NCCL debug flags
  5. Loss divergence:  reduce learning rate or rollback to last checkpoint
  6. Thermal throttle: reduce batch_size to lower power draw
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum, auto

from orchestrator.gpu_monitor import GPUMonitor, GPUSnapshot
from orchestrator.log_parser import LogParser, EventType, Severity

logger = logging.getLogger(__name__)


class ActionType(Enum):
    NO_OP = auto()
    REDUCE_BATCH_SIZE = auto()
    INCREASE_BATCH_SIZE = auto()
    ENABLE_ACTIVATION_CHECKPOINTING = auto()
    INCREASE_ZERO_STAGE = auto()
    DECREASE_ZERO_STAGE = auto()
    ENABLE_CPU_OFFLOAD = auto()
    REDUCE_LEARNING_RATE = auto()
    REQUEUE_JOB = auto()
    ROLLBACK_CHECKPOINT = auto()
    KILL_JOB = auto()


@dataclass
class ControlAction:
    """A decision made by the controller."""

    action_type: ActionType
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # higher = more urgent


@dataclass
class TrainingState:
    """Current mutable state of the training job, modified by the controller."""

    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    zero_stage: int = 2
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    fp16: bool = True
    current_step: int = 0
    max_steps: int = 50000
    num_oom_events: int = 0
    num_requeues: int = 0
    max_requeues: int = 5
    is_running: bool = False
    job_id: Optional[str] = None

    # Bounds
    min_batch_size: int = 4
    max_batch_size: int = 128
    min_zero_stage: int = 0
    max_zero_stage: int = 3

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "zero_stage": self.zero_stage,
            "activation_checkpointing": self.activation_checkpointing,
            "cpu_offload": self.cpu_offload,
            "fp16": self.fp16,
            "current_step": self.current_step,
            "num_oom_events": self.num_oom_events,
            "num_requeues": self.num_requeues,
        }


class RuleBasedController:
    """
    Autonomous rule-based controller for multi-GPU training.

    Consumes GPU snapshots and log events, produces ControlActions
    that the orchestrator applies to the running training job.
    """

    def __init__(
        self,
        gpu_monitor: GPUMonitor,
        log_parser: LogParser,
        state: Optional[TrainingState] = None,
        thresholds: Optional[dict] = None,
    ):
        self.gpu_monitor = gpu_monitor
        self.log_parser = log_parser
        self.state = state or TrainingState()

        # Configurable thresholds
        t = thresholds or {}
        self.memory_warn_pct = t.get("memory_warn_pct", 85.0)
        self.memory_critical_pct = t.get("memory_critical_pct", 93.0)
        self.memory_oom_pct = t.get("memory_oom_pct", 97.0)
        self.util_low_pct = t.get("util_low_pct", 30.0)
        self.util_target_pct = t.get("util_target_pct", 80.0)
        self.temp_warn_c = t.get("temp_warn_c", 80)
        self.temp_critical_c = t.get("temp_critical_c", 88)

        # Action history for logging / analysis
        self._action_history: List[ControlAction] = []

        # Cooldowns: prevent rapid-fire decisions
        self._last_action_time: Dict[ActionType, float] = {}
        self._cooldown_sec: Dict[ActionType, float] = {
            ActionType.REDUCE_BATCH_SIZE: 60,
            ActionType.INCREASE_BATCH_SIZE: 120,
            ActionType.ENABLE_ACTIVATION_CHECKPOINTING: 30,
            ActionType.INCREASE_ZERO_STAGE: 60,
            ActionType.DECREASE_ZERO_STAGE: 120,
            ActionType.ENABLE_CPU_OFFLOAD: 60,
            ActionType.REDUCE_LEARNING_RATE: 120,
            ActionType.REQUEUE_JOB: 120,
            ActionType.ROLLBACK_CHECKPOINT: 180,
            ActionType.KILL_JOB: 0,
        }

        # Track manual intervention count
        self.manual_interventions = 0
        self.auto_interventions = 0

    def _is_cooled_down(self, action_type: ActionType) -> bool:
        last = self._last_action_time.get(action_type, 0)
        cooldown = self._cooldown_sec.get(action_type, 60)
        return (time.time() - last) >= cooldown

    def _record_action(self, action: ControlAction):
        self._action_history.append(action)
        self._last_action_time[action.action_type] = time.time()
        self.auto_interventions += 1
        logger.info(
            f"CONTROLLER ACTION: {action.action_type.name} | "
            f"Reason: {action.reason} | Params: {action.params}"
        )

    def evaluate(self) -> List[ControlAction]:
        """
        Main decision loop. Evaluates all rules and returns a prioritized
        list of actions to take.
        """
        actions = []

        # 1. Check for OOM events in logs
        actions.extend(self._check_oom())

        # 2. Check for NCCL errors
        actions.extend(self._check_nccl())

        # 3. Check GPU memory pressure
        actions.extend(self._check_memory_pressure())

        # 4. Check GPU underutilization
        actions.extend(self._check_underutilization())

        # 5. Check thermal throttling
        actions.extend(self._check_thermal())

        # 6. Check loss divergence
        actions.extend(self._check_loss_divergence())

        # Sort by priority (highest first) and return
        actions.sort(key=lambda a: a.priority, reverse=True)

        for action in actions:
            self._record_action(action)

        return actions

    def _check_oom(self) -> List[ControlAction]:
        """Handle CUDA OOM errors with escalating responses."""
        actions = []

        if self.log_parser.oom_count <= self.state.num_oom_events:
            return actions

        # New OOM detected
        new_ooms = self.log_parser.oom_count - self.state.num_oom_events
        self.state.num_oom_events = self.log_parser.oom_count
        logger.warning(f"Detected {new_ooms} new OOM event(s)")

        # Escalation ladder:
        # Step 1: Enable activation checkpointing (if not already)
        if not self.state.activation_checkpointing:
            actions.append(ControlAction(
                action_type=ActionType.ENABLE_ACTIVATION_CHECKPOINTING,
                reason=f"OOM detected (count={self.log_parser.oom_count}); enabling activation checkpointing",
                priority=90,
            ))
            return actions

        # Step 2: Reduce batch size
        if self.state.batch_size > self.state.min_batch_size:
            new_bs = max(self.state.min_batch_size, self.state.batch_size // 2)
            actions.append(ControlAction(
                action_type=ActionType.REDUCE_BATCH_SIZE,
                reason=f"OOM detected with checkpointing on; reducing batch_size {self.state.batch_size} → {new_bs}",
                params={"new_batch_size": new_bs},
                priority=85,
            ))
            return actions

        # Step 3: Increase ZeRO stage
        if self.state.zero_stage < self.state.max_zero_stage:
            new_stage = self.state.zero_stage + 1
            actions.append(ControlAction(
                action_type=ActionType.INCREASE_ZERO_STAGE,
                reason=f"OOM with min batch & checkpointing; ZeRO {self.state.zero_stage} → {new_stage}",
                params={"new_zero_stage": new_stage},
                priority=80,
            ))
            return actions

        # Step 4: Enable CPU offload (ZeRO-3)
        if not self.state.cpu_offload and self.state.zero_stage == 3:
            actions.append(ControlAction(
                action_type=ActionType.ENABLE_CPU_OFFLOAD,
                reason="OOM at ZeRO-3; enabling CPU offload as last resort",
                priority=75,
            ))
            return actions

        # Step 5: Give up if max requeues exceeded
        if self.state.num_requeues >= self.state.max_requeues:
            actions.append(ControlAction(
                action_type=ActionType.KILL_JOB,
                reason=f"OOM persists after {self.state.max_requeues} requeues; giving up",
                priority=100,
            ))
        else:
            # Requeue with current (reduced) config
            actions.append(ControlAction(
                action_type=ActionType.REQUEUE_JOB,
                reason="OOM: requeuing job with adjusted parameters",
                params=self.state.to_dict(),
                priority=70,
            ))

        return actions

    def _check_nccl(self) -> List[ControlAction]:
        """Handle NCCL communication errors."""
        actions = []

        if self.log_parser.nccl_error_count == 0:
            return actions

        if not self._is_cooled_down(ActionType.REQUEUE_JOB):
            return actions

        if self.state.num_requeues >= self.state.max_requeues:
            actions.append(ControlAction(
                action_type=ActionType.KILL_JOB,
                reason=f"NCCL errors persist after {self.state.num_requeues} requeues",
                priority=95,
            ))
        else:
            actions.append(ControlAction(
                action_type=ActionType.REQUEUE_JOB,
                reason=f"NCCL error detected ({self.log_parser.nccl_error_count} total); requeuing with NCCL debug",
                params={
                    "env_overrides": {
                        "NCCL_DEBUG": "INFO",
                        "NCCL_SOCKET_IFNAME": "eth0",
                        "NCCL_IB_DISABLE": "1",
                    },
                    **self.state.to_dict(),
                },
                priority=88,
            ))

        return actions

    def _check_memory_pressure(self) -> List[ControlAction]:
        """Proactively reduce workload when GPU memory is near capacity."""
        actions = []

        histories = self.gpu_monitor.get_all_histories()
        for gpu_id, hist in histories.items():
            latest = hist.latest
            if not latest:
                continue

            # Critical: memory > 93% and rising
            if latest.memory_pct >= self.memory_critical_pct:
                if hist.memory_trend > 0.5:  # rising >0.5%/min
                    if not self.state.activation_checkpointing:
                        if self._is_cooled_down(ActionType.ENABLE_ACTIVATION_CHECKPOINTING):
                            actions.append(ControlAction(
                                action_type=ActionType.ENABLE_ACTIVATION_CHECKPOINTING,
                                reason=f"GPU {gpu_id} memory critical ({latest.memory_pct:.1f}%) and rising",
                                priority=70,
                            ))
                    elif self.state.batch_size > self.state.min_batch_size:
                        if self._is_cooled_down(ActionType.REDUCE_BATCH_SIZE):
                            new_bs = max(
                                self.state.min_batch_size,
                                int(self.state.batch_size * 0.75),
                            )
                            actions.append(ControlAction(
                                action_type=ActionType.REDUCE_BATCH_SIZE,
                                reason=f"GPU {gpu_id} memory critical ({latest.memory_pct:.1f}%); preemptive reduction",
                                params={"new_batch_size": new_bs},
                                priority=65,
                            ))

            # Warning: memory > 85%
            elif latest.memory_pct >= self.memory_warn_pct:
                if hist.memory_trend > 1.0:  # rapidly rising
                    logger.warning(
                        f"GPU {gpu_id} memory high ({latest.memory_pct:.1f}%) "
                        f"and rising ({hist.memory_trend:.1f}%/min)"
                    )

        return actions

    def _check_underutilization(self) -> List[ControlAction]:
        """Increase batch size if GPU is underutilized."""
        actions = []

        histories = self.gpu_monitor.get_all_histories()
        all_low = True
        all_safe_mem = True

        for gpu_id, hist in histories.items():
            if hist.avg_utilization > self.util_low_pct:
                all_low = False
            if hist.peak_memory_pct > self.memory_warn_pct:
                all_safe_mem = False

        if (
            all_low
            and all_safe_mem
            and self.state.batch_size < self.state.max_batch_size
            and self._is_cooled_down(ActionType.INCREASE_BATCH_SIZE)
        ):
            new_bs = min(self.state.max_batch_size, int(self.state.batch_size * 1.25))
            if new_bs != self.state.batch_size:
                actions.append(ControlAction(
                    action_type=ActionType.INCREASE_BATCH_SIZE,
                    reason=f"GPUs underutilised (avg util < {self.util_low_pct}%); "
                           f"batch_size {self.state.batch_size} → {new_bs}",
                    params={"new_batch_size": new_bs},
                    priority=20,
                ))

        return actions

    def _check_thermal(self) -> List[ControlAction]:
        """Reduce workload if GPUs are overheating."""
        actions = []

        snapshots = self.gpu_monitor.poll()
        for gpu_id, snap in snapshots.items():
            if snap.temperature_c >= self.temp_critical_c:
                if (
                    self.state.batch_size > self.state.min_batch_size
                    and self._is_cooled_down(ActionType.REDUCE_BATCH_SIZE)
                ):
                    new_bs = max(
                        self.state.min_batch_size,
                        int(self.state.batch_size * 0.75),
                    )
                    actions.append(ControlAction(
                        action_type=ActionType.REDUCE_BATCH_SIZE,
                        reason=f"GPU {gpu_id} overheating ({snap.temperature_c}°C); reducing batch size",
                        params={"new_batch_size": new_bs},
                        priority=60,
                    ))

        return actions

    def _check_loss_divergence(self) -> List[ControlAction]:
        """Handle diverging loss."""
        actions = []

        if not self.log_parser.is_loss_diverging():
            return actions

        if self._is_cooled_down(ActionType.REDUCE_LEARNING_RATE):
            new_lr = self.state.learning_rate * 0.5
            actions.append(ControlAction(
                action_type=ActionType.REDUCE_LEARNING_RATE,
                reason=f"Loss diverging; reducing LR {self.state.learning_rate:.2e} → {new_lr:.2e}",
                params={"new_lr": new_lr},
                priority=50,
            ))

        return actions

    def apply_action(self, action: ControlAction) -> bool:
        """
        Apply a ControlAction to the current TrainingState.
        Returns True if state was modified.
        """
        modified = False

        if action.action_type == ActionType.REDUCE_BATCH_SIZE:
            new_bs = action.params.get("new_batch_size", self.state.batch_size // 2)
            new_bs = max(self.state.min_batch_size, new_bs)
            if new_bs != self.state.batch_size:
                logger.info(f"Batch size: {self.state.batch_size} → {new_bs}")
                self.state.batch_size = new_bs
                modified = True

        elif action.action_type == ActionType.INCREASE_BATCH_SIZE:
            new_bs = action.params.get(
                "new_batch_size", int(self.state.batch_size * 1.25)
            )
            new_bs = min(self.state.max_batch_size, new_bs)
            if new_bs != self.state.batch_size:
                logger.info(f"Batch size: {self.state.batch_size} → {new_bs}")
                self.state.batch_size = new_bs
                modified = True

        elif action.action_type == ActionType.ENABLE_ACTIVATION_CHECKPOINTING:
            if not self.state.activation_checkpointing:
                logger.info("Enabling activation checkpointing")
                self.state.activation_checkpointing = True
                modified = True

        elif action.action_type == ActionType.INCREASE_ZERO_STAGE:
            new_stage = action.params.get(
                "new_zero_stage", self.state.zero_stage + 1
            )
            new_stage = min(self.state.max_zero_stage, new_stage)
            if new_stage != self.state.zero_stage:
                logger.info(f"ZeRO stage: {self.state.zero_stage} → {new_stage}")
                self.state.zero_stage = new_stage
                modified = True

        elif action.action_type == ActionType.DECREASE_ZERO_STAGE:
            new_stage = action.params.get(
                "new_zero_stage", self.state.zero_stage - 1
            )
            new_stage = max(self.state.min_zero_stage, new_stage)
            if new_stage != self.state.zero_stage:
                logger.info(f"ZeRO stage: {self.state.zero_stage} → {new_stage}")
                self.state.zero_stage = new_stage
                modified = True

        elif action.action_type == ActionType.ENABLE_CPU_OFFLOAD:
            if not self.state.cpu_offload:
                logger.info("Enabling CPU offload")
                self.state.cpu_offload = True
                modified = True

        elif action.action_type == ActionType.REDUCE_LEARNING_RATE:
            new_lr = action.params.get("new_lr", self.state.learning_rate * 0.5)
            if new_lr != self.state.learning_rate:
                logger.info(f"LR: {self.state.learning_rate:.2e} → {new_lr:.2e}")
                self.state.learning_rate = new_lr
                modified = True

        elif action.action_type == ActionType.REQUEUE_JOB:
            self.state.num_requeues += 1
            logger.info(f"Requeue #{self.state.num_requeues}")
            modified = True

        elif action.action_type == ActionType.KILL_JOB:
            logger.critical("Controller decided to KILL the job")
            self.state.is_running = False
            modified = True

        return modified

    @property
    def action_history(self) -> List[ControlAction]:
        return list(self._action_history)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Controller Summary",
            "=" * 60,
            f"  Auto interventions:   {self.auto_interventions}",
            f"  Manual interventions:  {self.manual_interventions}",
            f"  Total requeues:        {self.state.num_requeues}",
            f"  Current batch_size:    {self.state.batch_size}",
            f"  Current ZeRO stage:    {self.state.zero_stage}",
            f"  Act. checkpointing:    {self.state.activation_checkpointing}",
            f"  CPU offload:           {self.state.cpu_offload}",
            f"  Learning rate:         {self.state.learning_rate:.2e}",
            "  Recent actions:",
        ]
        for action in self._action_history[-10:]:
            lines.append(
                f"    [{action.action_type.name}] {action.reason}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_state(self, path: str):
        """Save controller state to JSON."""
        data = {
            "training_state": self.state.to_dict(),
            "auto_interventions": self.auto_interventions,
            "manual_interventions": self.manual_interventions,
            "action_history": [
                {
                    "action": a.action_type.name,
                    "reason": a.reason,
                    "params": a.params,
                    "timestamp": a.timestamp,
                }
                for a in self._action_history
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Controller state saved to {path}")
