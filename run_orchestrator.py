#!/usr/bin/env python
"""
Entry point for the Autonomous Multi-GPU Experiment Orchestrator.

Usage:
    # Run with default config
    python run_orchestrator.py

    # Run with custom config
    python run_orchestrator.py --config configs/default_config.yaml

    # Specify GPU count
    python run_orchestrator.py --num_gpus 4
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.orchestrator import main

if __name__ == "__main__":
    main()
