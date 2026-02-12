"""
300M Parameter Transformer Encoder Training Script
====================================================
Trains a 300M-parameter transformer encoder with DeepSpeed.

Architecture (≈300M params):
  - hidden_size:       1024
  - num_layers:        18
  - num_heads:         16
  - intermediate_size: 4096
  - vocab_size:        30522
  - max_seq_length:    512

Supports:
  - FP16 mixed precision
  - Activation checkpointing (gradient checkpointing)
  - DeepSpeed ZeRO stages 0-3
  - Multi-GPU via torchrun / SLURM
  - Checkpoint save/resume
"""

import os
import sys
import math
import time
import json
import random
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("trainer")


# ========================================================================
# Model: Transformer Encoder (300M params)
# ========================================================================

class TransformerEncoderModel(nn.Module):
    """
    BERT-style transformer encoder.

    With hidden=1024, layers=18, heads=16, intermediate=4096, vocab=30522:
    Total params ≈ 300M
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 1024,
        num_layers: int = 18,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        encoded = self.encoder(embeddings)
        logits = self.mlm_head(encoded)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ========================================================================
# Synthetic Dataset (MLM-style)
# ========================================================================

class SyntheticMLMDataset(Dataset):
    """
    Generates synthetic masked language modeling data.
    Used for benchmarking without needing a real corpus.
    """

    def __init__(
        self,
        num_samples: int = 100000,
        seq_length: int = 512,
        vocab_size: int = 30522,
        mask_prob: float = 0.15,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random token IDs (skip special tokens 0-4)
        input_ids = [self.rng.randint(5, self.vocab_size - 1) for _ in range(self.seq_length)]

        # Create MLM labels (-100 = ignore, otherwise the original token)
        labels = [-100] * self.seq_length
        for i in range(self.seq_length):
            if self.rng.random() < self.mask_prob:
                labels[i] = input_ids[i]
                # 80% [MASK], 10% random, 10% keep
                r = self.rng.random()
                if r < 0.8:
                    input_ids[i] = 103  # [MASK] token
                elif r < 0.9:
                    input_ids[i] = self.rng.randint(5, self.vocab_size - 1)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ========================================================================
# Training Loop
# ========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="300M Transformer Encoder Training")

    # Model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--intermediate_size", type=int, default=4096)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=100000)

    # Mixed precision & memory
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_activation_checkpointing", action="store_true")

    # DeepSpeed
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)

    # I/O
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume_from", type=str, default=None)

    # Benchmark mode
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_warmup_steps", type=int, default=20)
    parser.add_argument("--benchmark_measure_steps", type=int, default=100)

    # DeepSpeed launcher adds these
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False
    if args.no_activation_checkpointing:
        args.activation_checkpointing = False

    return args


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def create_deepspeed_config(args) -> dict:
    """Create DeepSpeed config from args."""
    config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": args.log_interval,
        "wall_clock_breakdown": False,
    }

    if args.fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }

    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": args.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
        },
    }

    config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": args.learning_rate,
            "warmup_num_steps": args.warmup_steps,
            "total_num_steps": args.max_steps,
        },
    }

    zero_config = {
        "stage": args.zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    }

    if args.cpu_offload and args.zero_stage >= 3:
        zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    config["zero_optimization"] = zero_config

    if args.activation_checkpointing:
        config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        }

    return config


def train(args):
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    if is_main:
        logger.info(f"World size: {world_size}")
        logger.info(f"Args: {vars(args)}")

    # Seed
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # Model
    model = TransformerEncoderModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
    )

    if is_main:
        total_params = model.count_parameters()
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Activation checkpointing
    if args.activation_checkpointing:
        if hasattr(torch.utils, 'checkpoint'):
            # Wrap encoder layers with gradient checkpointing
            for layer in model.encoder.layers:
                layer._orig_forward = layer.forward
                def make_ckpt_forward(mod):
                    def ckpt_forward(*a, **kw):
                        return torch.utils.checkpoint.checkpoint(mod._orig_forward, *a, use_reentrant=False, **kw)
                    return ckpt_forward
                layer.forward = make_ckpt_forward(layer)
            if is_main:
                logger.info("Activation checkpointing enabled")

    # Dataset
    dataset = SyntheticMLMDataset(
        num_samples=args.num_samples,
        seq_length=args.max_seq_length,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # DeepSpeed or DDP
    # Use DeepSpeed when ZeRO > 0 (DeepSpeed handles its own dist init)
    use_deepspeed = DEEPSPEED_AVAILABLE and args.zero_stage > 0

    if use_deepspeed:
        ds_config = create_deepspeed_config(args)

        if args.deepspeed_config:
            with open(args.deepspeed_config) as f:
                ds_config = json.load(f)

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        device = model_engine.local_rank
        if is_main:
            logger.info(f"DeepSpeed initialized (ZeRO stage {args.zero_stage})")
    else:
        device = local_rank
        model = model.to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_steps
        )
        model_engine = None

    # Resume from checkpoint
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        if use_deepspeed:
            _, client_state = model_engine.load_checkpoint(args.resume_from)
            if client_state:
                start_step = client_state.get("step", 0)
        else:
            ckpt = torch.load(args.resume_from, map_location=f"cuda:{local_rank}")
            m = model.module if hasattr(model, "module") else model
            m.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", 0)
        if is_main:
            logger.info(f"Resumed from step {start_step}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Training
    if is_main:
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)

    # Track peak memory
    torch.cuda.reset_peak_memory_stats(device)

    global_step = start_step
    total_loss = 0.0
    total_tokens = 0
    train_start = time.time()
    step_start = time.time()

    # Benchmark mode
    if args.benchmark:
        args.max_steps = args.benchmark_warmup_steps + args.benchmark_measure_steps

    benchmark_times = []
    benchmark_samples = []

    data_iter = iter(dataloader)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler:
                sampler.set_epoch(global_step)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if use_deepspeed:
            outputs = model_engine(input_ids, labels=labels)
            loss = outputs["loss"]
            model_engine.backward(loss)
            model_engine.step()
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        global_step += 1
        total_loss += loss.item()
        total_tokens += input_ids.numel()

        # Benchmark timing
        if args.benchmark and global_step > args.benchmark_warmup_steps:
            step_time = time.time() - step_start
            benchmark_times.append(step_time)
            benchmark_samples.append(args.batch_size * world_size)

        step_start = time.time()

        # Logging
        if is_main and global_step % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - train_start
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            samples_per_sec = (global_step * args.batch_size * world_size) / elapsed if elapsed > 0 else 0

            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            current_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)

            current_lr = 0
            if use_deepspeed:
                current_lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else args.learning_rate
            elif scheduler:
                current_lr = scheduler.get_last_lr()[0]

            logger.info(
                f"step={global_step} | loss={avg_loss:.4f} | "
                f"lr={current_lr:.2e} | "
                f"{samples_per_sec:.1f} samples/sec | "
                f"{tokens_per_sec:.0f} tokens/sec | "
                f"mem={current_mem:.2f}/{peak_mem:.2f} GiB"
            )

            total_loss = 0.0

        # Save checkpoint
        if is_main and global_step % args.save_interval == 0 and not args.benchmark:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)

            if use_deepspeed:
                model_engine.save_checkpoint(ckpt_dir, client_state={"step": global_step})
            else:
                m = model.module if hasattr(model, "module") else model
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": m.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(ckpt_dir, "model.pt"),
                )

            logger.info(f"Saved checkpoint at step {global_step}")

    # Final stats
    total_time = time.time() - train_start
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    if is_main:
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"  Total steps:    {global_step}")
        logger.info(f"  Total time:     {total_time:.1f}s")
        logger.info(f"  Peak GPU mem:   {peak_mem_gb:.2f} GiB")
        logger.info(f"  World size:     {world_size}")
        logger.info("=" * 60)

    # Benchmark results
    if args.benchmark and is_main:
        if benchmark_times:
            avg_step_time = sum(benchmark_times) / len(benchmark_times)
            total_benchmark_samples = sum(benchmark_samples)
            total_benchmark_time = sum(benchmark_times)
            throughput = total_benchmark_samples / total_benchmark_time

            results = {
                "world_size": world_size,
                "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": args.batch_size * world_size,
                "zero_stage": args.zero_stage,
                "fp16": args.fp16,
                "activation_checkpointing": args.activation_checkpointing,
                "avg_step_time_sec": avg_step_time,
                "throughput_samples_per_sec": throughput,
                "peak_gpu_memory_gib": peak_mem_gb,
                "total_params": model.count_parameters() if not use_deepspeed else sum(p.numel() for p in model_engine.module.parameters()),
                "measure_steps": len(benchmark_times),
            }

            logger.info("BENCHMARK RESULTS:")
            for k, v in results.items():
                logger.info(f"  {k}: {v}")

            results_path = os.path.join(args.output_dir, "benchmark_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to {results_path}")

    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
