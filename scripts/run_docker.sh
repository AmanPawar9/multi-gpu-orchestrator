#!/bin/bash
# ============================================================
# Docker launch script for the orchestrator
# ============================================================

docker run --rm -it \
    --runtime=nvidia \
    --gpus all \
    --shm-size=16g \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:24.01-py3 \
    bash -c "pip install deepspeed pyyaml psutil nvidia-ml-py && python run_orchestrator.py --config configs/default_config.yaml"
