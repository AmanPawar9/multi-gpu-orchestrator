#!/usr/bin/env python
"""
Entry point for running benchmarks.

Usage:
    python run_benchmark.py
    python run_benchmark.py --gpus 4 --output_dir ./benchmark_outputs
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.run_benchmarks import main

if __name__ == "__main__":
    main()
