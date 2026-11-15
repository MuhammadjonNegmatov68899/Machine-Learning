#!/usr/bin/env python3
"""
CLI driver for Lab 6 Task 1: YAML-Driven Training Pipeline.
Usage:
    python lab06/scripts/run_pipeline.py [--config lab06/configs/train_config.yaml] [--device auto]
"""

import argparse
import sys
import os

# Add repo root to python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from lab06.src.pipeline import train_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run Lab 6 YAML-driven training pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="lab06/configs/train_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('auto', 'mps', 'cpu', 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print(" Lab 6: Reproducible YAML-Driven Pipeline")
    print(f" Config: {args.config}")
    print("=" * 65)

    results = train_pipeline(args.config, override_device=args.device)
    print("\nPipeline execution completed successfully.")
    print(f"Final Test Loss: {results['test_loss']:.4f}")
    print(f"Final Test Acc:  {results['test_acc']:.2f}%")
    print(f"Best Checkpoint: {results['best_checkpoint_path']}")


if __name__ == "__main__":
    main()
