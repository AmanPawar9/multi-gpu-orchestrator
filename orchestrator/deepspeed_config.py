"""
DeepSpeed Configuration Generator
===================================
Dynamically generates DeepSpeed JSON configs based on the current
TrainingState. Called by the orchestrator before each job submission.
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def generate_deepspeed_config(
    zero_stage: int = 2,
    fp16: bool = True,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    activation_checkpointing: bool = True,
    cpu_offload: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """
    Generate a DeepSpeed configuration dictionary.

    Supports ZeRO stages 0-3, FP16, activation checkpointing,
    and CPU offloading.
    """
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
    }

    # FP16
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    else:
        config["fp16"] = {"enabled": False}

    # Optimizer
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    }

    # Scheduler
    config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": learning_rate,
            "warmup_num_steps": 1000,
            "total_num_steps": 50000,
        },
    }

    # ZeRO optimization
    zero_config = {
        "stage": zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    }

    if zero_stage >= 2:
        zero_config["round_robin_gradients"] = True

    if zero_stage >= 3:
        zero_config["stage3_max_live_parameters"] = 1e9
        zero_config["stage3_max_reuse_distance"] = 1e9
        zero_config["stage3_prefetch_bucket_size"] = 5e8
        zero_config["stage3_param_persistence_threshold"] = 1e6
        zero_config["sub_group_size"] = 1e12
        zero_config["stage3_gather_16bit_weights_on_model_save"] = True

    # CPU offloading (ZeRO-3 only)
    if cpu_offload and zero_stage >= 3:
        zero_config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": False,
        }

    config["zero_optimization"] = zero_config

    # Activation checkpointing
    if activation_checkpointing:
        config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        }

    # Communication
    config["comms_logger"] = {"enabled": False}

    # Save to file if path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"DeepSpeed config saved to {output_path}")

    return config


def generate_from_state(state, output_path: Optional[str] = None) -> dict:
    """Generate DeepSpeed config from a TrainingState object."""
    return generate_deepspeed_config(
        zero_stage=state.zero_stage,
        fp16=state.fp16,
        batch_size=state.batch_size,
        gradient_accumulation_steps=state.gradient_accumulation_steps,
        learning_rate=state.learning_rate,
        activation_checkpointing=state.activation_checkpointing,
        cpu_offload=state.cpu_offload,
        output_path=output_path,
    )
