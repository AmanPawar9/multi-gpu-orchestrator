"""
SLURM / Docker Interface
=========================
Abstracted job launcher that supports:
  - SLURM (sbatch / scancel / squeue / scontrol)
  - Docker (docker run with --gpus, --shm-size)
  - Local (direct subprocess launch for single-node dev)

Provides:
  - Job submission with dynamic parameter injection
  - Job status querying
  - Automatic requeue with parameter modification
  - Job cancellation
"""

import os
import json
import shutil
import signal
import subprocess
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)


class LaunchMode(Enum):
    LOCAL = auto()
    SLURM = auto()
    DOCKER = auto()


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    OOM_KILLED = "OOM_KILLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobInfo:
    job_id: str
    status: JobStatus
    launch_mode: LaunchMode
    submit_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    node_list: Optional[str] = None
    gpu_count: int = 1
    params: Dict[str, Any] = field(default_factory=dict)


class JobLauncher:
    """
    Unified interface for launching training jobs across SLURM, Docker, or local.

    The orchestrator calls `submit()` to start a training job and `requeue()` to
    restart it with modified parameters. The launcher generates the appropriate
    command/script and manages the lifecycle.
    """

    def __init__(self, config: dict):
        self.config = config
        self.mode = self._detect_mode()
        self._active_jobs: Dict[str, JobInfo] = {}
        self._local_processes: Dict[str, subprocess.Popen] = {}
        self._job_counter = 0

        logger.info(f"Job launcher initialized in {self.mode.name} mode")

    def _detect_mode(self) -> LaunchMode:
        """Auto-detect available launch mode."""
        slurm_cfg = self.config.get("slurm", {})
        docker_cfg = self.config.get("docker", {})

        if slurm_cfg.get("enabled") and shutil.which("sbatch"):
            return LaunchMode.SLURM
        elif docker_cfg.get("enabled") and shutil.which("docker"):
            return LaunchMode.DOCKER
        else:
            return LaunchMode.LOCAL

    def submit(
        self,
        script_path: str,
        training_args: Dict[str, Any],
        env_overrides: Optional[Dict[str, str]] = None,
        gpu_count: Optional[int] = None,
    ) -> JobInfo:
        """Submit a training job."""
        if self.mode == LaunchMode.SLURM:
            return self._submit_slurm(script_path, training_args, env_overrides, gpu_count)
        elif self.mode == LaunchMode.DOCKER:
            return self._submit_docker(script_path, training_args, env_overrides, gpu_count)
        else:
            return self._submit_local(script_path, training_args, env_overrides, gpu_count)

    def _submit_local(
        self,
        script_path: str,
        training_args: Dict[str, Any],
        env_overrides: Optional[Dict[str, str]] = None,
        gpu_count: Optional[int] = None,
    ) -> JobInfo:
        """Launch training as a local subprocess using torchrun."""
        self._job_counter += 1
        job_id = f"local_{self._job_counter}_{int(time.time())}"

        n_gpu = gpu_count or training_args.get("num_gpus", 1)

        # Build command
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_gpu}",
            "--master_port", str(29500 + self._job_counter),
            script_path,
        ]

        # Add training args as CLI flags
        for key, value in training_args.items():
            if key == "num_gpus":
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        # Environment
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        log_dir = self.config.get("experiment", {}).get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{job_id}.log")

        logger.info(f"Launching local job {job_id}: {' '.join(cmd)}")
        logger.info(f"Log file: {log_file}")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
            )

        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.RUNNING,
            launch_mode=LaunchMode.LOCAL,
            submit_time=time.time(),
            start_time=time.time(),
            gpu_count=n_gpu,
            params={**training_args, "log_file": log_file},
        )

        self._active_jobs[job_id] = job_info
        self._local_processes[job_id] = proc

        return job_info

    def _submit_slurm(
        self,
        script_path: str,
        training_args: Dict[str, Any],
        env_overrides: Optional[Dict[str, str]] = None,
        gpu_count: Optional[int] = None,
    ) -> JobInfo:
        """Submit via SLURM sbatch."""
        self._job_counter += 1
        slurm_cfg = self.config.get("slurm", {})
        n_gpu = gpu_count or slurm_cfg.get("gpus_per_node", 4)

        # Build training args string
        args_str = ""
        for key, value in training_args.items():
            if key == "num_gpus":
                continue
            if isinstance(value, bool):
                if value:
                    args_str += f" --{key}"
            else:
                args_str += f" --{key} {value}"

        # Build env export string
        env_str = ""
        if env_overrides:
            for k, v in env_overrides.items():
                env_str += f"export {k}={v}\n"

        log_dir = self.config.get("experiment", {}).get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)

        # Generate SLURM script
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={self.config.get('experiment', {}).get('name', 'train')}
#SBATCH --partition={slurm_cfg.get('partition', 'gpu')}
#SBATCH --nodes={slurm_cfg.get('nodes', 1)}
#SBATCH --gpus-per-node={n_gpu}
#SBATCH --cpus-per-gpu={slurm_cfg.get('cpus_per_gpu', 4)}
#SBATCH --mem-per-gpu={slurm_cfg.get('mem_per_gpu', '32G')}
#SBATCH --time={slurm_cfg.get('time_limit', '24:00:00')}
#SBATCH --output={log_dir}/slurm_%j.log
#SBATCH --error={log_dir}/slurm_%j.err
#SBATCH --requeue
{"#SBATCH --account=" + slurm_cfg['account'] if slurm_cfg.get('account') else ""}
{"#SBATCH --qos=" + slurm_cfg['qos'] if slurm_cfg.get('qos') else ""}

{env_str}

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

srun torchrun \\
    --nproc_per_node={n_gpu} \\
    --nnodes=$SLURM_NNODES \\
    --node_rank=$SLURM_PROCID \\
    --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \\
    --master_port=29500 \\
    {script_path} {args_str}

echo "Job $SLURM_JOB_ID finished at $(date) with exit code $?"
"""
        sbatch_path = os.path.join(log_dir, f"job_{self._job_counter}.sbatch")
        with open(sbatch_path, "w") as f:
            f.write(sbatch_script)

        try:
            result = subprocess.run(
                ["sbatch", sbatch_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse job ID from "Submitted batch job 12345"
            slurm_job_id = result.stdout.strip().split()[-1]
        except (subprocess.TimeoutExpired, FileNotFoundError, IndexError) as e:
            logger.error(f"SLURM submission failed: {e}")
            slurm_job_id = f"failed_{self._job_counter}"

        job_info = JobInfo(
            job_id=slurm_job_id,
            status=JobStatus.PENDING,
            launch_mode=LaunchMode.SLURM,
            submit_time=time.time(),
            gpu_count=n_gpu,
            params={**training_args, "sbatch_path": sbatch_path},
        )
        self._active_jobs[slurm_job_id] = job_info

        logger.info(f"SLURM job submitted: {slurm_job_id}")
        return job_info

    def _submit_docker(
        self,
        script_path: str,
        training_args: Dict[str, Any],
        env_overrides: Optional[Dict[str, str]] = None,
        gpu_count: Optional[int] = None,
    ) -> JobInfo:
        """Launch training in a Docker container."""
        self._job_counter += 1
        job_id = f"docker_{self._job_counter}_{int(time.time())}"
        docker_cfg = self.config.get("docker", {})
        n_gpu = gpu_count or training_args.get("num_gpus", 1)

        # Build training args string
        args_parts = []
        for key, value in training_args.items():
            if key == "num_gpus":
                continue
            if isinstance(value, bool):
                if value:
                    args_parts.append(f"--{key}")
            else:
                args_parts.append(f"--{key}")
                args_parts.append(str(value))

        args_str = " ".join(args_parts)

        cmd = [
            "docker", "run",
            "--rm",
            f"--runtime={docker_cfg.get('runtime', 'nvidia')}",
            f"--gpus={n_gpu}",
            f"--shm-size={docker_cfg.get('shm_size', '16g')}",
            "--name", job_id,
            "-v", f"{os.getcwd()}:/workspace",
            "-w", "/workspace",
        ]

        # Add env overrides
        if env_overrides:
            for k, v in env_overrides.items():
                cmd.extend(["-e", f"{k}={v}"])

        image = docker_cfg.get("image", "nvcr.io/nvidia/pytorch:24.01-py3")
        cmd.append(image)

        # Training command inside container
        cmd.extend([
            "torchrun",
            f"--nproc_per_node={n_gpu}",
            script_path,
        ])
        cmd.extend(args_parts)

        log_dir = self.config.get("experiment", {}).get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{job_id}.log")

        logger.info(f"Launching Docker job {job_id}")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.RUNNING,
            launch_mode=LaunchMode.DOCKER,
            submit_time=time.time(),
            start_time=time.time(),
            gpu_count=n_gpu,
            params={**training_args, "log_file": log_file},
        )

        self._active_jobs[job_id] = job_info
        self._local_processes[job_id] = proc

        return job_info

    def get_status(self, job_id: str) -> JobStatus:
        """Check current job status."""
        job_info = self._active_jobs.get(job_id)
        if not job_info:
            return JobStatus.UNKNOWN

        if job_info.launch_mode == LaunchMode.LOCAL:
            return self._get_local_status(job_id)
        elif job_info.launch_mode == LaunchMode.SLURM:
            return self._get_slurm_status(job_id)
        elif job_info.launch_mode == LaunchMode.DOCKER:
            return self._get_docker_status(job_id)

        return JobStatus.UNKNOWN

    def _get_local_status(self, job_id: str) -> JobStatus:
        proc = self._local_processes.get(job_id)
        if not proc:
            return JobStatus.UNKNOWN

        poll = proc.poll()
        if poll is None:
            return JobStatus.RUNNING
        elif poll == 0:
            self._active_jobs[job_id].status = JobStatus.COMPLETED
            self._active_jobs[job_id].end_time = time.time()
            self._active_jobs[job_id].exit_code = 0
            return JobStatus.COMPLETED
        elif poll == -9 or poll == 137:  # SIGKILL / OOM killer
            self._active_jobs[job_id].status = JobStatus.OOM_KILLED
            self._active_jobs[job_id].end_time = time.time()
            self._active_jobs[job_id].exit_code = poll
            return JobStatus.OOM_KILLED
        else:
            self._active_jobs[job_id].status = JobStatus.FAILED
            self._active_jobs[job_id].end_time = time.time()
            self._active_jobs[job_id].exit_code = poll
            return JobStatus.FAILED

    def _get_slurm_status(self, job_id: str) -> JobStatus:
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            status_str = result.stdout.strip().upper()

            status_map = {
                "PENDING": JobStatus.PENDING,
                "RUNNING": JobStatus.RUNNING,
                "COMPLETING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "TIMEOUT": JobStatus.TIMEOUT,
                "CANCELLED": JobStatus.CANCELLED,
                "OUT_OF_MEMORY": JobStatus.OOM_KILLED,
            }
            status = status_map.get(status_str, JobStatus.UNKNOWN)

            # If not in queue, check sacct
            if not status_str:
                status = self._check_sacct(job_id)

            self._active_jobs[job_id].status = status
            return status

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return JobStatus.UNKNOWN

    def _check_sacct(self, job_id: str) -> JobStatus:
        """Check completed job status via sacct."""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-o", "State", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            states = result.stdout.strip().split("\n")
            if states:
                state = states[0].strip().upper()
                if "COMPLETED" in state:
                    return JobStatus.COMPLETED
                elif "FAILED" in state:
                    return JobStatus.FAILED
                elif "OUT_OF_ME" in state:
                    return JobStatus.OOM_KILLED
                elif "TIMEOUT" in state:
                    return JobStatus.TIMEOUT
                elif "CANCEL" in state:
                    return JobStatus.CANCELLED
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return JobStatus.UNKNOWN

    def _get_docker_status(self, job_id: str) -> JobStatus:
        proc = self._local_processes.get(job_id)
        if proc:
            return self._get_local_status(job_id)
        return JobStatus.UNKNOWN

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        job_info = self._active_jobs.get(job_id)
        if not job_info:
            logger.warning(f"Job {job_id} not found")
            return False

        success = False

        if job_info.launch_mode == LaunchMode.LOCAL:
            proc = self._local_processes.get(job_id)
            if proc and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
                success = True
        elif job_info.launch_mode == LaunchMode.SLURM:
            try:
                subprocess.run(["scancel", job_id], timeout=10)
                success = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        elif job_info.launch_mode == LaunchMode.DOCKER:
            try:
                subprocess.run(["docker", "stop", job_id], timeout=30)
                success = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        if success:
            job_info.status = JobStatus.CANCELLED
            job_info.end_time = time.time()
            logger.info(f"Job {job_id} cancelled")

        return success

    def requeue(
        self,
        job_id: str,
        script_path: str,
        new_args: Dict[str, Any],
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> Optional[JobInfo]:
        """Cancel current job and resubmit with new parameters."""
        old_job = self._active_jobs.get(job_id)

        # Cancel running job
        self.cancel(job_id)

        # Small delay to allow GPU memory cleanup
        time.sleep(5)

        gpu_count = old_job.gpu_count if old_job else None

        logger.info(f"Requeuing job {job_id} with new params: {new_args}")
        return self.submit(script_path, new_args, env_overrides, gpu_count)

    def get_log_path(self, job_id: str) -> Optional[str]:
        """Get the log file path for a job."""
        job_info = self._active_jobs.get(job_id)
        if job_info:
            return job_info.params.get("log_file")
        return None

    @property
    def active_jobs(self) -> Dict[str, JobInfo]:
        return dict(self._active_jobs)

    def summary(self) -> str:
        lines = ["=" * 60, "Job Launcher Summary", "=" * 60]
        lines.append(f"  Mode: {self.mode.name}")
        lines.append(f"  Active jobs: {len(self._active_jobs)}")
        for jid, info in self._active_jobs.items():
            elapsed = ""
            if info.start_time:
                dt = (info.end_time or time.time()) - info.start_time
                elapsed = f" ({dt:.0f}s)"
            lines.append(
                f"    {jid}: {info.status.value} | "
                f"{info.gpu_count} GPU(s){elapsed}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
