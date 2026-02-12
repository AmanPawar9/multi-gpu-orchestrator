"""
Autonomous Multi-GPU Experiment Orchestrator (Agentic Trainer)

Monitors multi-GPU training metrics and automatically adapts job parameters
(batch size, checkpoint policy, ZeRO stage) via SLURM/Docker.
"""

__version__ = "1.0.0"
