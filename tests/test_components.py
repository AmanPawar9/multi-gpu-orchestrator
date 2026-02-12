"""
Unit tests for orchestrator components.
Tests GPU monitor, log parser, controller, and DeepSpeed config generation.
"""

import os
import sys
import time
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.gpu_monitor import GPUMonitor, GPUSnapshot, GPUHistory
from orchestrator.log_parser import LogParser, EventType, Severity
from orchestrator.controller import RuleBasedController, TrainingState, ActionType
from orchestrator.deepspeed_config import generate_deepspeed_config


# ========================================================================
# GPU Monitor Tests
# ========================================================================

class TestGPUMonitor:
    def test_init(self):
        monitor = GPUMonitor()
        assert monitor.gpu_count >= 1

    def test_poll_returns_snapshots(self):
        monitor = GPUMonitor()
        snapshots = monitor.poll()
        assert len(snapshots) >= 1
        for gpu_id, snap in snapshots.items():
            assert isinstance(snap, GPUSnapshot)
            assert snap.memory_total_mb > 0
            assert 0 <= snap.memory_pct <= 100

    def test_history_tracking(self):
        monitor = GPUMonitor()
        for _ in range(5):
            monitor.poll()
        hist = monitor.get_history(0)
        assert hist is not None
        assert len(hist.snapshots) == 5

    def test_history_peak_and_avg(self):
        hist = GPUHistory()
        for i in range(5):
            hist.add(GPUSnapshot(
                gpu_id=0, timestamp=time.time(), name="test",
                memory_used_mb=1000 + i * 100,
                memory_total_mb=8192,
                memory_pct=50 + i * 5,
                utilization_pct=70,
                temperature_c=60,
                power_draw_w=150,
                power_limit_w=250,
                clock_sm_mhz=1500,
                clock_mem_mhz=1000,
            ))
        assert hist.peak_memory_pct == 70.0
        assert hist.avg_memory_pct == 60.0

    def test_summary(self):
        monitor = GPUMonitor()
        summary = monitor.summary()
        assert "GPU Monitor Summary" in summary


# ========================================================================
# Log Parser Tests
# ========================================================================

class TestLogParser:
    def test_detect_oom(self):
        parser = LogParser()
        event = parser.parse_line("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
        assert event is not None
        assert event.event_type == EventType.OOM
        assert event.severity == Severity.FATAL
        assert parser.oom_count == 1

    def test_detect_nccl_error(self):
        parser = LogParser()
        event = parser.parse_line("NCCL error in: ncclCommInitRank")
        assert event is not None
        assert event.event_type == EventType.NCCL_ERROR
        assert parser.nccl_error_count == 1

    def test_detect_nccl_timeout(self):
        parser = LogParser()
        event = parser.parse_line("Watchdog caught collective operation timeout: NCCL")
        assert event is not None
        assert event.event_type == EventType.NCCL_TIMEOUT

    def test_detect_training_step(self):
        parser = LogParser()
        event = parser.parse_line("step=100 | loss=2.3456 | lr=1.00e-04 | 150.5 samples/sec")
        assert event is not None
        assert event.event_type == EventType.TRAINING_STEP
        assert event.metadata["step"] == 100
        assert abs(event.metadata["loss"] - 2.3456) < 1e-4
        assert abs(event.metadata["throughput"] - 150.5) < 1e-1

    def test_detect_nan_loss(self):
        parser = LogParser()
        event = parser.parse_line("Step 500: loss=nan")
        assert event is not None
        assert event.event_type == EventType.NAN_LOSS
        assert event.severity == Severity.CRITICAL

    def test_detect_grad_overflow(self):
        parser = LogParser()
        event = parser.parse_line("[DeepSpeed] Gradient overflow detected, skipped step")
        assert event is not None
        assert event.event_type == EventType.GRAD_OVERFLOW

    def test_detect_checkpoint(self):
        parser = LogParser()
        event = parser.parse_line("Saved checkpoint at step 1000")
        assert event is not None
        assert event.event_type == EventType.CHECKPOINT_SAVED

    def test_no_match(self):
        parser = LogParser()
        event = parser.parse_line("Just a regular log line with nothing special")
        assert event is None

    def test_empty_line(self):
        parser = LogParser()
        event = parser.parse_line("")
        assert event is None

    def test_loss_diverging(self):
        parser = LogParser()
        # Simulate increasing loss
        for i in range(30):
            parser.parse_line(f"step={i} | loss={1.0 + i * 0.5}")
        assert parser.is_loss_diverging(window=20, threshold=1.5)

    def test_loss_not_diverging(self):
        parser = LogParser()
        # Simulate stable loss
        for i in range(30):
            parser.parse_line(f"step={i} | loss={2.0}")
        assert not parser.is_loss_diverging()

    def test_parse_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("step=1 | loss=3.0\n")
            f.write("step=2 | loss=2.5\n")
            f.write("RuntimeError: CUDA out of memory\n")
            f.write("step=3 | loss=2.0\n")
            f.name

        try:
            parser = LogParser()
            events = parser.parse_file(f.name)
            assert len(events) == 4  # 3 steps + 1 OOM
            assert parser.oom_count == 1
        finally:
            os.unlink(f.name)

    def test_callback(self):
        parser = LogParser()
        events_received = []
        parser.register_callback(lambda e: events_received.append(e))
        parser.parse_line("CUDA out of memory")
        assert len(events_received) == 1

    def test_summary(self):
        parser = LogParser()
        parser.parse_line("step=10 | loss=2.5")
        summary = parser.summary()
        assert "Log Parser Summary" in summary


# ========================================================================
# Controller Tests
# ========================================================================

class TestController:
    def _make_controller(self, **state_kwargs):
        monitor = GPUMonitor()
        parser = LogParser()
        state = TrainingState(**state_kwargs)
        return RuleBasedController(monitor, parser, state)

    def test_oom_enables_checkpointing(self):
        controller = self._make_controller(activation_checkpointing=False)
        controller.log_parser.parse_line("CUDA out of memory")
        actions = controller.evaluate()
        assert any(a.action_type == ActionType.ENABLE_ACTIVATION_CHECKPOINTING for a in actions)

    def test_oom_reduces_batch_size(self):
        controller = self._make_controller(
            batch_size=32,
            activation_checkpointing=True,
        )
        controller.log_parser.parse_line("CUDA out of memory")
        actions = controller.evaluate()
        assert any(a.action_type == ActionType.REDUCE_BATCH_SIZE for a in actions)

    def test_oom_escalation_to_zero(self):
        controller = self._make_controller(
            batch_size=4,
            min_batch_size=4,
            activation_checkpointing=True,
            zero_stage=1,
        )
        controller.log_parser.parse_line("CUDA out of memory")
        actions = controller.evaluate()
        assert any(a.action_type == ActionType.INCREASE_ZERO_STAGE for a in actions)

    def test_nccl_triggers_requeue(self):
        controller = self._make_controller()
        controller.log_parser.parse_line("NCCL error in: ncclCommInitRank")
        actions = controller.evaluate()
        assert any(a.action_type == ActionType.REQUEUE_JOB for a in actions)

    def test_loss_divergence_reduces_lr(self):
        controller = self._make_controller(learning_rate=1e-4)
        # First 15 steps: low loss; last 15 steps: very high loss (>2x ratio)
        for i in range(15):
            controller.log_parser.parse_line(f"step={i} | loss=1.0")
        for i in range(15, 30):
            controller.log_parser.parse_line(f"step={i} | loss=10.0")
        actions = controller.evaluate()
        assert any(a.action_type == ActionType.REDUCE_LEARNING_RATE for a in actions)

    def test_apply_reduce_batch_size(self):
        controller = self._make_controller(batch_size=32)
        from orchestrator.controller import ControlAction
        action = ControlAction(
            action_type=ActionType.REDUCE_BATCH_SIZE,
            reason="test",
            params={"new_batch_size": 16},
        )
        modified = controller.apply_action(action)
        assert modified
        assert controller.state.batch_size == 16

    def test_apply_enable_checkpointing(self):
        controller = self._make_controller(activation_checkpointing=False)
        from orchestrator.controller import ControlAction
        action = ControlAction(
            action_type=ActionType.ENABLE_ACTIVATION_CHECKPOINTING,
            reason="test",
        )
        modified = controller.apply_action(action)
        assert modified
        assert controller.state.activation_checkpointing is True

    def test_apply_increase_zero_stage(self):
        controller = self._make_controller(zero_stage=1)
        from orchestrator.controller import ControlAction
        action = ControlAction(
            action_type=ActionType.INCREASE_ZERO_STAGE,
            reason="test",
            params={"new_zero_stage": 2},
        )
        modified = controller.apply_action(action)
        assert modified
        assert controller.state.zero_stage == 2

    def test_max_requeues_triggers_kill(self):
        controller = self._make_controller(
            batch_size=4,
            min_batch_size=4,
            activation_checkpointing=True,
            zero_stage=3,
            max_zero_stage=3,
            cpu_offload=True,
            num_requeues=5,
            max_requeues=5,
        )
        controller.log_parser.parse_line("CUDA out of memory")
        actions = controller.evaluate()
        assert any(a.action_type == ActionType.KILL_JOB for a in actions)

    def test_no_actions_when_stable(self):
        controller = self._make_controller()
        # No errors, no pressure
        actions = controller.evaluate()
        # Should have no critical actions
        assert not any(a.action_type in (
            ActionType.KILL_JOB,
            ActionType.REQUEUE_JOB,
            ActionType.REDUCE_BATCH_SIZE,
        ) for a in actions)

    def test_summary(self):
        controller = self._make_controller()
        summary = controller.summary()
        assert "Controller Summary" in summary

    def test_save_state(self):
        controller = self._make_controller()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            controller.save_state(path)
            with open(path) as f:
                data = json.load(f)
            assert "training_state" in data
            assert "action_history" in data
        finally:
            os.unlink(path)


# ========================================================================
# DeepSpeed Config Tests
# ========================================================================

class TestDeepSpeedConfig:
    def test_zero_stage_2_fp16(self):
        config = generate_deepspeed_config(zero_stage=2, fp16=True)
        assert config["zero_optimization"]["stage"] == 2
        assert config["fp16"]["enabled"] is True

    def test_zero_stage_0(self):
        config = generate_deepspeed_config(zero_stage=0)
        assert config["zero_optimization"]["stage"] == 0

    def test_zero_stage_3_with_offload(self):
        config = generate_deepspeed_config(zero_stage=3, cpu_offload=True)
        assert config["zero_optimization"]["stage"] == 3
        assert "offload_param" in config["zero_optimization"]
        assert "offload_optimizer" in config["zero_optimization"]

    def test_activation_checkpointing(self):
        config = generate_deepspeed_config(activation_checkpointing=True)
        assert "activation_checkpointing" in config

    def test_no_activation_checkpointing(self):
        config = generate_deepspeed_config(activation_checkpointing=False)
        assert "activation_checkpointing" not in config

    def test_save_to_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            config = generate_deepspeed_config(output_path=path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["zero_optimization"]["stage"] == config["zero_optimization"]["stage"]
        finally:
            os.unlink(path)

    def test_batch_size_params(self):
        config = generate_deepspeed_config(batch_size=64, gradient_accumulation_steps=8)
        assert config["train_micro_batch_size_per_gpu"] == 64
        assert config["gradient_accumulation_steps"] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
