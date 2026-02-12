"""
Main Orchestrator
==================
The central coordination loop that ties everything together:
  1. Launches training jobs via SLURM/Docker/local
  2. Monitors GPU metrics (GPUMonitor)
  3. Parses training logs in real-time (LogParser)
  4. Runs the rule-based controller to detect issues & adapt params
  5. Applies controller decisions (requeue, adjust batch size, etc.)
  6. Logs all decisions and metrics for post-hoc analysis
"""

import os
import sys
import time
import json
import yaml
import signal
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from orchestrator.gpu_monitor import GPUMonitor
from orchestrator.log_parser import LogParser, EventType, Severity
from orchestrator.controller import RuleBasedController, TrainingState, ActionType
from orchestrator.slurm_interface import JobLauncher, JobStatus
from orchestrator.deepspeed_config import generate_from_state

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("orchestrator")


class ExperimentOrchestrator:
    """
    Autonomous Multi-GPU Experiment Orchestrator.

    Monitors training, detects issues, and automatically adapts
    parameters to keep training stable with minimal manual intervention.
    """

    def __init__(self, config_path: str):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        exp_cfg = self.config.get("experiment", {})
        self.experiment_name = exp_cfg.get("name", "experiment")
        self.output_dir = exp_cfg.get("output_dir", "./outputs")
        self.log_dir = exp_cfg.get("log_dir", "./logs")
        self.poll_interval = exp_cfg.get("poll_interval_sec", 10)
        self.max_retries = exp_cfg.get("max_retries", 5)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Add file handler for orchestrator log
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, "orchestrator.log")
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        # Initialize components
        self.gpu_monitor = GPUMonitor()
        self.log_parser = LogParser()
        self.training_state = self._init_training_state()
        self.controller = RuleBasedController(
            gpu_monitor=self.gpu_monitor,
            log_parser=self.log_parser,
            state=self.training_state,
            thresholds=self.config.get("gpu_thresholds", {}),
        )
        self.job_launcher = JobLauncher(self.config)

        # State
        self._current_job_id: Optional[str] = None
        self._stop_event = threading.Event()
        self._running = False
        self._metrics_log: list = []

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(f"Orchestrator initialized: {self.experiment_name}")
        logger.info(f"Output: {self.output_dir} | Logs: {self.log_dir}")

    def _init_training_state(self) -> TrainingState:
        """Initialize TrainingState from config."""
        train_cfg = self.config.get("training", {})
        ds_cfg = self.config.get("deepspeed", {})

        return TrainingState(
            batch_size=train_cfg.get("initial_batch_size", 32),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=train_cfg.get("learning_rate", 1e-4),
            zero_stage=ds_cfg.get("initial_zero_stage", 2),
            activation_checkpointing=ds_cfg.get("activation_checkpointing", True),
            cpu_offload=ds_cfg.get("cpu_offload", False),
            fp16=train_cfg.get("fp16", True),
            max_steps=train_cfg.get("max_steps", 50000),
            max_requeues=self.config.get("experiment", {}).get("max_retries", 5),
            min_batch_size=train_cfg.get("min_batch_size", 4),
            max_batch_size=train_cfg.get("max_batch_size", 128),
            min_zero_stage=ds_cfg.get("min_zero_stage", 0),
            max_zero_stage=ds_cfg.get("max_zero_stage", 3),
        )

    def _build_training_args(self) -> Dict[str, Any]:
        """Build training script arguments from current state."""
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})
        state = self.training_state

        # Generate DeepSpeed config
        ds_config_path = os.path.join(self.output_dir, "ds_config.json")
        generate_from_state(state, output_path=ds_config_path)

        args = {
            # Model
            "hidden_size": model_cfg.get("hidden_size", 1024),
            "num_layers": model_cfg.get("num_layers", 18),
            "num_heads": model_cfg.get("num_heads", 16),
            "intermediate_size": model_cfg.get("intermediate_size", 4096),
            "vocab_size": model_cfg.get("vocab_size", 30522),
            "max_seq_length": model_cfg.get("max_seq_length", 512),
            # Training
            "batch_size": state.batch_size,
            "gradient_accumulation_steps": state.gradient_accumulation_steps,
            "learning_rate": state.learning_rate,
            "warmup_steps": train_cfg.get("warmup_steps", 1000),
            "max_steps": state.max_steps,
            "seed": self.config.get("experiment", {}).get("seed", 42),
            # DeepSpeed
            "zero_stage": state.zero_stage,
            "deepspeed_config": ds_config_path,
            # I/O
            "output_dir": self.output_dir,
            "log_interval": train_cfg.get("log_interval", 50),
            "save_interval": train_cfg.get("save_interval", 1000),
        }

        if state.fp16:
            args["fp16"] = True
        if state.activation_checkpointing:
            args["activation_checkpointing"] = True
        if state.cpu_offload:
            args["cpu_offload"] = True

        # Resume from latest checkpoint
        latest_ckpt = self._find_latest_checkpoint()
        if latest_ckpt:
            args["resume_from"] = latest_ckpt
            logger.info(f"Will resume from checkpoint: {latest_ckpt}")

        return args

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint directory."""
        ckpt_dirs = sorted(
            Path(self.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if ckpt_dirs:
            return str(ckpt_dirs[-1])
        return None

    def _get_script_path(self) -> str:
        """Get the path to the training script."""
        base_dir = Path(__file__).parent.parent
        return str(base_dir / "scripts" / "train_transformer.py")

    def start(self, num_gpus: Optional[int] = None):
        """
        Start the orchestrator: launch training and begin monitoring.
        """
        self._running = True
        logger.info("=" * 60)
        logger.info("AUTONOMOUS EXPERIMENT ORCHESTRATOR STARTING")
        logger.info("=" * 60)

        # Initial GPU status
        logger.info(self.gpu_monitor.summary())

        # Determine GPU count
        if num_gpus is None:
            num_gpus = self.gpu_monitor.gpu_count
        logger.info(f"Using {num_gpus} GPU(s)")

        # Launch initial job
        script_path = self._get_script_path()
        training_args = self._build_training_args()
        training_args["num_gpus"] = num_gpus

        job_info = self.job_launcher.submit(
            script_path=script_path,
            training_args=training_args,
            gpu_count=num_gpus,
        )
        self._current_job_id = job_info.job_id
        self.training_state.is_running = True
        self.training_state.job_id = job_info.job_id

        # Start log parsing
        log_path = self.job_launcher.get_log_path(job_info.job_id)
        if log_path:
            self.log_parser.start_tailing(log_path)

        # Start GPU monitoring
        self.gpu_monitor.start_background(interval_sec=self.poll_interval)

        # Enter monitoring loop
        logger.info(f"Job launched: {job_info.job_id}")
        logger.info(f"Entering monitoring loop (poll interval: {self.poll_interval}s)")
        self._monitoring_loop(num_gpus)

    def _monitoring_loop(self, num_gpus: int):
        """
        Main monitoring loop. Polls GPU metrics, checks logs,
        runs the controller, and applies decisions.
        """
        iteration = 0
        while not self._stop_event.is_set() and self._running:
            try:
                iteration += 1

                # 1. Check job status
                status = self.job_launcher.get_status(self._current_job_id)

                if status == JobStatus.COMPLETED:
                    logger.info("Training job completed successfully!")
                    self._running = False
                    break

                if status in (JobStatus.FAILED, JobStatus.OOM_KILLED, JobStatus.TIMEOUT):
                    logger.warning(f"Job ended with status: {status.value}")
                    self._handle_job_failure(status, num_gpus)
                    if not self._running:
                        break
                    continue

                # 2. Collect metrics
                gpu_snapshots = self.gpu_monitor.poll()
                self._log_metrics(iteration, gpu_snapshots)

                # 3. Run controller evaluation
                actions = self.controller.evaluate()

                # 4. Apply actions
                for action in actions:
                    self._apply_action(action, num_gpus)

                # 5. Periodic status log
                if iteration % 6 == 0:  # ~every minute
                    self._log_status()

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)

            self._stop_event.wait(self.poll_interval)

        self._shutdown()

    def _apply_action(self, action, num_gpus: int):
        """Apply a controller action."""
        modified = self.controller.apply_action(action)

        if not modified:
            return

        if action.action_type == ActionType.KILL_JOB:
            logger.critical("Controller issued KILL. Stopping orchestrator.")
            if self._current_job_id:
                self.job_launcher.cancel(self._current_job_id)
            self._running = False
            return

        if action.action_type == ActionType.REQUEUE_JOB:
            self._requeue(num_gpus, action.params.get("env_overrides"))
            return

        # For parameter changes that need a restart
        if action.action_type in (
            ActionType.REDUCE_BATCH_SIZE,
            ActionType.INCREASE_BATCH_SIZE,
            ActionType.ENABLE_ACTIVATION_CHECKPOINTING,
            ActionType.INCREASE_ZERO_STAGE,
            ActionType.DECREASE_ZERO_STAGE,
            ActionType.ENABLE_CPU_OFFLOAD,
        ):
            logger.info(f"Parameter changed by controller — requeuing job")
            self._requeue(num_gpus)

    def _requeue(self, num_gpus: int, env_overrides: Optional[dict] = None):
        """Requeue the training job with updated parameters."""
        if self.training_state.num_requeues >= self.max_retries:
            logger.critical(
                f"Max requeues ({self.max_retries}) reached. Stopping."
            )
            self._running = False
            return

        # Stop log tailing
        self.log_parser.stop_tailing()
        self.log_parser.reset()

        # Build new args
        training_args = self._build_training_args()
        training_args["num_gpus"] = num_gpus

        script_path = self._get_script_path()

        # Requeue
        new_job = self.job_launcher.requeue(
            job_id=self._current_job_id,
            script_path=script_path,
            new_args=training_args,
            env_overrides=env_overrides,
        )

        if new_job:
            self._current_job_id = new_job.job_id
            self.training_state.job_id = new_job.job_id

            # Restart log tailing
            log_path = self.job_launcher.get_log_path(new_job.job_id)
            if log_path:
                self.log_parser.start_tailing(log_path)

            logger.info(f"Job requeued as: {new_job.job_id}")
        else:
            logger.error("Requeue failed")
            self._running = False

    def _handle_job_failure(self, status: JobStatus, num_gpus: int):
        """Handle a failed training job."""
        if status == JobStatus.OOM_KILLED:
            logger.warning("Job was OOM-killed by the OS/SLURM")
            self.training_state.num_oom_events += 1

        if self.training_state.num_requeues >= self.max_retries:
            logger.critical("Max requeues reached after failure. Stopping.")
            self._running = False
            return

        # Let the controller decide what to adjust
        actions = self.controller.evaluate()
        for action in actions:
            self.controller.apply_action(action)

        # Requeue with adjusted params
        self.training_state.num_requeues += 1
        self._requeue(num_gpus)

    def _log_metrics(self, iteration: int, gpu_snapshots: dict):
        """Log metrics for post-hoc analysis."""
        entry = {
            "iteration": iteration,
            "timestamp": time.time(),
            "training_state": self.training_state.to_dict(),
            "gpu_metrics": {},
        }
        for gpu_id, snap in gpu_snapshots.items():
            entry["gpu_metrics"][gpu_id] = {
                "memory_pct": snap.memory_pct,
                "memory_used_mb": snap.memory_used_mb,
                "utilization_pct": snap.utilization_pct,
                "temperature_c": snap.temperature_c,
                "power_w": snap.power_draw_w,
            }

        if self.log_parser.latest_loss is not None:
            entry["loss"] = self.log_parser.latest_loss
            entry["step"] = self.log_parser.latest_step

        self._metrics_log.append(entry)

    def _log_status(self):
        """Log periodic status summary."""
        logger.info("-" * 40)
        logger.info(self.gpu_monitor.summary())
        logger.info(self.log_parser.summary())
        logger.info(self.controller.summary())
        logger.info(self.job_launcher.summary())
        logger.info("-" * 40)

    def _handle_signal(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self._stop_event.set()
        self._running = False

    def _shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down orchestrator...")

        # Stop monitoring
        self.gpu_monitor.stop_background()
        self.log_parser.stop_tailing()

        # Save final state
        self._save_report()

        # Save controller state
        controller_state_path = os.path.join(self.output_dir, "controller_state.json")
        self.controller.save_state(controller_state_path)

        # Shutdown GPU monitor
        self.gpu_monitor.shutdown()

        logger.info("Orchestrator shutdown complete.")

    def _save_report(self):
        """Save experiment report."""
        report = {
            "experiment": self.experiment_name,
            "final_state": self.training_state.to_dict(),
            "auto_interventions": self.controller.auto_interventions,
            "manual_interventions": self.controller.manual_interventions,
            "total_requeues": self.training_state.num_requeues,
            "action_history": [
                {
                    "action": a.action_type.name,
                    "reason": a.reason,
                    "timestamp": a.timestamp,
                }
                for a in self.controller.action_history
            ],
            "metrics_log_entries": len(self._metrics_log),
        }

        report_path = os.path.join(self.output_dir, "experiment_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Experiment report saved to {report_path}")

        # Save full metrics log
        metrics_path = os.path.join(self.output_dir, "metrics_log.json")
        with open(metrics_path, "w") as f:
            json.dump(self._metrics_log, f, indent=2)
        logger.info(f"Metrics log saved to {metrics_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Experiment Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs (auto-detected if not specified)",
    )
    args = parser.parse_args()

    orchestrator = ExperimentOrchestrator(args.config)
    orchestrator.start(num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
