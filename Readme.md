Autonomous Multi-GPU Experiment Orchestrator
Path: ~/ap/multi-gpu-orchestrator

Project Structure
File	Purpose
orchestrator/gpu_monitor.py	Real-time GPU monitoring via pynvml (memory, util, temp, power, trends)
orchestrator/log_parser.py	Log parser with regex-based detection of OOM, NCCL errors, NaN loss, grad overflow
orchestrator/controller.py	Rule-based agentic controller with escalation ladder (batch size → checkpointing → ZeRO stage → CPU offload → requeue → kill)
orchestrator/slurm_interface.py	Unified job launcher supporting SLURM, Docker, and local (torchrun) modes
orchestrator/deepspeed_config.py	Dynamic DeepSpeed JSON config generation based on current training state
orchestrator/orchestrator.py	Main orchestrator loop: monitor → detect → decide → apply → requeue
scripts/train_transformer.py	300M param transformer encoder (290.8M confirmed) with DeepSpeed ZeRO + FP16 + activation checkpointing
benchmarks/run_benchmarks.py	Scaling efficiency and memory optimization benchmarks
configs/default_config.yaml	Full configuration with thresholds, model, training, SLURM, Docker settings
tests/test_components.py	38 unit tests — all passing
Key Features Implemented
Agentic Control — Rule-based controller autonomously handles:

GPU OOMs (escalation: checkpointing → batch reduction → ZeRO upgrade → CPU offload)
NCCL errors (auto-requeue with debug flags)
Loss divergence (automatic LR reduction)
Thermal throttling (workload reduction)
GPU underutilization (batch size increase)
300M Transformer Encoder — hidden=1024, layers=18, heads=16, ffn=4096 (290.8M params verified)

Memory Optimization — FP16 + activation checkpointing + ZeRO-2 (peak 3.59 GiB on 8GB GPU)

Multi-backend — SLURM (sbatch/scancel/scontrol), Docker, or local torchrun

How to Run

cd ~/ap/multi-gpu-orchestrator
conda activate sae

# Run orchestrator (auto-detects GPUs, launches training, monitors + adapts)
python run_orchestrator.py --config configs/default_config.yaml

# Run benchmarks (memory optimization + scaling efficiency)
python run_benchmark.py --output_dir ./benchmark_outputs

# Run tests
python -m pytest tests/ -v

# Direct training (standalone)
torchrun --nproc_per_node=1 scripts/train_transformer.py --fp16 --activation_checkpointing --zero_stage 2 --batch_size 4 --benchmark