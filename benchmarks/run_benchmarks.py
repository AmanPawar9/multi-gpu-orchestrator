"""
Benchmarking & Scaling Efficiency Measurement
===============================================
Runs systematic benchmarks to measure:
  1. Single-GPU baseline throughput
  2. Multi-GPU scaling efficiency (1, 2, 4 GPUs)
  3. Memory savings from FP16 + activation checkpointing + ZeRO-2
  4. Throughput impact of different ZeRO stages

Produces a JSON report and terminal summary showing:
  - Throughput (samples/sec) per config
  - Peak GPU memory per config
  - Scaling efficiency (e.g., 3.6x on 4 GPUs)
  - Memory reduction percentages
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("benchmark")


@dataclass
class BenchmarkConfig:
    """A single benchmark configuration."""
    name: str
    num_gpus: int = 1
    batch_size: int = 16
    fp16: bool = True
    activation_checkpointing: bool = True
    zero_stage: int = 2
    warmup_steps: int = 10
    measure_steps: int = 50
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    num_gpus: int
    batch_size: int
    fp16: bool
    activation_checkpointing: bool
    zero_stage: int
    throughput_samples_per_sec: float = 0.0
    peak_gpu_memory_gib: float = 0.0
    avg_step_time_sec: float = 0.0
    total_params: int = 0
    success: bool = True
    error: str = ""


def get_script_path() -> str:
    return str(Path(__file__).parent.parent / "scripts" / "train_transformer.py")


def run_single_benchmark(config: BenchmarkConfig, output_dir: str) -> BenchmarkResult:
    """Run a single benchmark configuration and parse results."""
    logger.info(f"{'='*60}")
    logger.info(f"Running benchmark: {config.name}")
    logger.info(f"  GPUs: {config.num_gpus} | BS: {config.batch_size} | "
                f"FP16: {config.fp16} | ActCkpt: {config.activation_checkpointing} | "
                f"ZeRO: {config.zero_stage}")
    logger.info(f"{'='*60}")

    script_path = get_script_path()
    bench_output_dir = os.path.join(output_dir, config.name.replace(" ", "_"))
    os.makedirs(bench_output_dir, exist_ok=True)

    # Use a smaller model for benchmarking to save time
    # But keep the architecture to demonstrate the technique
    cmd = [
        "torchrun",
        f"--nproc_per_node={config.num_gpus}",
        "--master_port", str(29500 + hash(config.name) % 1000),
        script_path,
        "--benchmark",
        "--batch_size", str(config.batch_size),
        "--zero_stage", str(config.zero_stage),
        "--benchmark_warmup_steps", str(config.warmup_steps),
        "--benchmark_measure_steps", str(config.measure_steps),
        "--output_dir", bench_output_dir,
        "--log_interval", "10",
        # Use a smaller model for faster benchmarking
        "--hidden_size", str(config.extra_args.get("hidden_size", 1024)),
        "--num_layers", str(config.extra_args.get("num_layers", 18)),
        "--num_heads", str(config.extra_args.get("num_heads", 16)),
        "--intermediate_size", str(config.extra_args.get("intermediate_size", 4096)),
        "--max_seq_length", str(config.extra_args.get("max_seq_length", 512)),
        "--num_samples", "10000",
    ]

    if config.fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--no_fp16")

    if config.activation_checkpointing:
        cmd.append("--activation_checkpointing")
    else:
        cmd.append("--no_activation_checkpointing")

    # Run benchmark
    log_path = os.path.join(bench_output_dir, "benchmark.log")
    result = BenchmarkResult(
        config_name=config.name,
        num_gpus=config.num_gpus,
        batch_size=config.batch_size,
        fp16=config.fp16,
        activation_checkpointing=config.activation_checkpointing,
        zero_stage=config.zero_stage,
    )

    try:
        logger.info(f"Command: {' '.join(cmd)}")
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=600,  # 10 min timeout per benchmark
            )

        if proc.returncode != 0:
            result.success = False
            result.error = f"Process exited with code {proc.returncode}"
            logger.error(f"Benchmark failed: {result.error}")
            # Try to read log for error details
            try:
                with open(log_path) as f:
                    last_lines = f.readlines()[-20:]
                    result.error += "\n" + "".join(last_lines)
            except Exception:
                pass
            return result

        # Parse results from saved JSON
        results_json = os.path.join(bench_output_dir, "benchmark_results.json")
        if os.path.exists(results_json):
            with open(results_json) as f:
                data = json.load(f)
            result.throughput_samples_per_sec = data.get("throughput_samples_per_sec", 0)
            result.peak_gpu_memory_gib = data.get("peak_gpu_memory_gib", 0)
            result.avg_step_time_sec = data.get("avg_step_time_sec", 0)
            result.total_params = data.get("total_params", 0)
            logger.info(f"  Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
            logger.info(f"  Peak GPU memory: {result.peak_gpu_memory_gib:.2f} GiB")
        else:
            # Parse from log
            result = _parse_log_for_results(log_path, result)

    except subprocess.TimeoutExpired:
        result.success = False
        result.error = "Benchmark timed out (600s)"
        logger.error(result.error)
    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error(f"Benchmark error: {e}")

    return result


def _parse_log_for_results(log_path: str, result: BenchmarkResult) -> BenchmarkResult:
    """Fallback: parse benchmark results from log file."""
    import re
    try:
        with open(log_path) as f:
            content = f.read()

        # Look for throughput
        tp_match = re.search(r"throughput_samples_per_sec:\s*([0-9.]+)", content)
        if tp_match:
            result.throughput_samples_per_sec = float(tp_match.group(1))

        mem_match = re.search(r"peak_gpu_memory_gib:\s*([0-9.]+)", content)
        if mem_match:
            result.peak_gpu_memory_gib = float(mem_match.group(1))

        time_match = re.search(r"avg_step_time_sec:\s*([0-9.]+)", content)
        if time_match:
            result.avg_step_time_sec = float(time_match.group(1))

        param_match = re.search(r"total_params:\s*(\d+)", content)
        if param_match:
            result.total_params = int(param_match.group(1))

    except Exception as e:
        logger.warning(f"Log parse error: {e}")

    return result


def run_all_benchmarks(
    output_dir: str = "./benchmark_outputs",
    available_gpus: int = 1,
) -> List[BenchmarkResult]:
    """
    Run the complete benchmark suite.

    Configurations:
    1. Baseline: FP32, no checkpointing, ZeRO-0, 1 GPU
    2. FP16 only: 1 GPU
    3. FP16 + activation checkpointing: 1 GPU
    4. FP16 + activation checkpointing + ZeRO-2: 1 GPU
    5. Multi-GPU scaling: 1, 2, 4 GPUs with best config
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use smaller batch for FP32 baseline (more memory hungry)
    base_bs = 8
    opt_bs = 16  # Can use bigger batch with FP16+checkpointing

    configs = [
        # Memory optimization benchmarks (single GPU)
        BenchmarkConfig(
            name="baseline_fp32_no_ckpt_zero0",
            num_gpus=1,
            batch_size=base_bs,
            fp16=False,
            activation_checkpointing=False,
            zero_stage=0,
        ),
        BenchmarkConfig(
            name="fp16_no_ckpt_zero0",
            num_gpus=1,
            batch_size=base_bs,
            fp16=True,
            activation_checkpointing=False,
            zero_stage=0,
        ),
        BenchmarkConfig(
            name="fp16_ckpt_zero0",
            num_gpus=1,
            batch_size=opt_bs,
            fp16=True,
            activation_checkpointing=True,
            zero_stage=0,
        ),
        BenchmarkConfig(
            name="fp16_ckpt_zero2",
            num_gpus=1,
            batch_size=opt_bs,
            fp16=True,
            activation_checkpointing=True,
            zero_stage=2,
        ),
    ]

    # Multi-GPU scaling benchmarks
    for n_gpu in [1, 2, 4]:
        if n_gpu <= available_gpus:
            configs.append(BenchmarkConfig(
                name=f"scaling_{n_gpu}gpu_fp16_ckpt_zero2",
                num_gpus=n_gpu,
                batch_size=opt_bs,
                fp16=True,
                activation_checkpointing=True,
                zero_stage=2,
            ))

    results = []
    for config in configs:
        if config.num_gpus > available_gpus:
            logger.warning(
                f"Skipping {config.name}: requires {config.num_gpus} GPUs, "
                f"only {available_gpus} available"
            )
            continue

        result = run_single_benchmark(config, output_dir)
        results.append(result)

        # Brief cooldown between benchmarks
        time.sleep(3)

    # Generate report
    generate_report(results, output_dir)

    return results


def generate_report(results: List[BenchmarkResult], output_dir: str):
    """Generate a comprehensive benchmark report."""

    # Save raw results
    raw_path = os.path.join(output_dir, "benchmark_results_all.json")
    with open(raw_path, "w") as f:
        json.dump(
            [
                {
                    "name": r.config_name,
                    "num_gpus": r.num_gpus,
                    "batch_size": r.batch_size,
                    "fp16": r.fp16,
                    "activation_checkpointing": r.activation_checkpointing,
                    "zero_stage": r.zero_stage,
                    "throughput": r.throughput_samples_per_sec,
                    "peak_memory_gib": r.peak_gpu_memory_gib,
                    "avg_step_time": r.avg_step_time_sec,
                    "total_params": r.total_params,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
            f,
            indent=2,
        )

    # Terminal report
    print("\n" + "=" * 80)
    print("BENCHMARK REPORT — Autonomous Multi-GPU Experiment Orchestrator")
    print("=" * 80)

    # Memory optimization comparison
    print("\n--- Memory Optimization (Single GPU) ---")
    print(f"{'Config':<35} {'Throughput':>12} {'Peak Mem':>10} {'Mem Savings':>12}")
    print("-" * 75)

    baseline_mem = None
    for r in results:
        if not r.success:
            print(f"{r.config_name:<35} {'FAILED':>12} {'':>10} {'':>12}")
            continue

        if r.config_name.startswith("baseline"):
            baseline_mem = r.peak_gpu_memory_gib

        mem_savings = ""
        if baseline_mem and baseline_mem > 0 and r.peak_gpu_memory_gib > 0:
            pct = ((baseline_mem - r.peak_gpu_memory_gib) / baseline_mem) * 100
            if pct > 0:
                mem_savings = f"{pct:.1f}%"

        if r.num_gpus == 1 and not r.config_name.startswith("scaling"):
            print(
                f"{r.config_name:<35} "
                f"{r.throughput_samples_per_sec:>10.2f}/s "
                f"{r.peak_gpu_memory_gib:>8.2f}G "
                f"{mem_savings:>12}"
            )

    # Scaling efficiency
    print("\n--- Multi-GPU Scaling ---")
    print(f"{'Config':<40} {'GPUs':>5} {'Throughput':>12} {'Scaling':>10}")
    print("-" * 70)

    single_gpu_throughput = None
    for r in results:
        if not r.success:
            continue
        if r.config_name.startswith("scaling"):
            if r.num_gpus == 1:
                single_gpu_throughput = r.throughput_samples_per_sec

            scaling = ""
            if single_gpu_throughput and single_gpu_throughput > 0:
                ratio = r.throughput_samples_per_sec / single_gpu_throughput
                efficiency = (ratio / r.num_gpus) * 100
                scaling = f"{ratio:.2f}x ({efficiency:.0f}%)"

            print(
                f"{r.config_name:<40} "
                f"{r.num_gpus:>5} "
                f"{r.throughput_samples_per_sec:>10.2f}/s "
                f"{scaling:>10}"
            )

    print("\n" + "=" * 80)
    print(f"Full results saved to: {raw_path}")
    print("=" * 80 + "\n")


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run orchestrator benchmarks")
    parser.add_argument("--output_dir", default="./benchmark_outputs")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of available GPUs (auto-detected)")
    args = parser.parse_args()

    if args.gpus is None:
        args.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.gpus == 0:
        logger.error("No GPUs detected. Benchmarks require at least 1 GPU.")
        sys.exit(1)

    logger.info(f"Available GPUs: {args.gpus}")
    run_all_benchmarks(output_dir=args.output_dir, available_gpus=args.gpus)


if __name__ == "__main__":
    main()
