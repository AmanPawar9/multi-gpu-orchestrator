# Autonomous Multi-GPU Experiment Orchestrator

An agentic, self-healing framework for distributed Deep Learning training.

Overview

This project implements an autonomous orchestrator that wraps around standard PyTorch/DeepSpeed training workflows. Unlike static training scripts, this orchestrator actively monitors hardware telemetry and training logs in real-time. It utilizes a rule-based agentic controller to autonomously detect failures (OOM, divergence, NCCL errors) and intervene by adjusting configuration parameters dynamically—without human intervention.

## Key Features

## Agentic Control & Self-Healing

The core controller.py implements a sophisticated escalation ladder to handle training failures autonomously:

GPU OOM Handling: Automatically iterates through mitigation strategies:

Reduce Batch Size

Enable Activation Checkpointing

Upgrade ZeRO Stage (1 → 2 → 3)

Enable CPU Offloading

Crash Recovery: Detects NCCL errors and auto-requeues jobs with specific debug flags.

Loss Stability: Detects divergence (NaN loss) and automatically triggers Learning Rate reduction.

Thermal Management: Detects thermal throttling and reduces workload intensity.

Efficiency: Detects GPU underutilization and safely increases batch sizes.

## 300M Transformer Architecture

The repository includes a verified implementation of a Transformer Encoder optimized for this orchestration:

Params: 290.8M (Verified)

Architecture: Hidden=1024, Layers=18, Heads=16, FFN=4096

Optimization: DeepSpeed ZeRO-2 + FP16 + Activation Checkpointing (Peak memory: 3.59 GiB on 8GB GPU).

## Multi-Backend Support

SLURM: Native integration for sbatch, scancel, and scontrol.

Docker: Containerized training support.

Local: Standalone torchrun for development and debugging.

## Project Structure

├── orchestrator/
│   ├── orchestrator.py      # Main Loop: Monitor → Detect → Decide → Apply → Requeue
│   ├── controller.py        # Agentic Logic: The escalation ladder and decision making
│   ├── gpu_monitor.py       # Pynvml wrapper: Real-time memory, util, temp, power tracking
│   ├── log_parser.py        # Regex detection: OOM, NCCL errors, NaN loss, grad overflow
│   ├── deepspeed_config.py  # Dynamic JSON config generation based on current state
│   └── slurm_interface.py   # Unified launcher (SLURM/Docker/Local)
├── scripts/
│   └── train_transformer.py # 300M param transformer model definition & training loop
├── configs/
│   └── default_config.yaml  # Central config: thresholds, model params, backend settings
├── benchmarks/
│   └── run_benchmarks.py    # Tools for scaling efficiency and memory profiling
└── tests/
    └── test_components.py   # 38 unit tests covering all core modules


## Getting Started

Prerequisites

Ensure you have a CUDA-capable environment.

# Clone the repository
git clone [https://github.com/AmanPawar9/multi-gpu-orchestrator.git](https://github.com/AmanPawar9/multi-gpu-orchestrator.git)
cd multi-gpu-orchestrator

# Activate environment
conda activate env


1. Run the Autonomous Orchestrator

This is the main mode of operation. The orchestrator will auto-detect available GPUs, launch the training job, and begin the monitoring loop.

python run_orchestrator.py --config configs/default_config.yaml


If an OOM occurs, the orchestrator will kill the job, adjust the DeepSpeed config (e.g., enable CPU offload), and restart training automatically.

2. Benchmarking

Run scaling efficiency and memory optimization benchmarks to profile your hardware.

python run_benchmark.py --output_dir ./benchmark_outputs


3. Manual / Standalone Training

For debugging model code without the orchestrator overhead, use torchrun directly.

torchrun --nproc_per_node=1 scripts/train_transformer.py \
    --fp16 \
    --activation_checkpointing \
    --zero_stage 2 \
    --batch_size 4 \
    --benchmark


4. Run Tests

Verify system integrity.

python -m pytest tests/ -v


## Configuration

The system is controlled via configs/default_config.yaml. Key sections include:

escalation_strategies: Define the order of operations for OOM mitigation.

thresholds: Set temperature limits (e.g., 80°C) and GPU utilization targets.

backend: Switch between slurm, docker, or local.

## Performance Notes

Memory Efficiency: With ZeRO-2 and Activation Checkpointing, the 300M model fits comfortably on consumer-grade GPUs (8GB VRAM).

Scalability: The log parser effectively identifies gradient overflow, ensuring training stability even at lower precision (FP16).