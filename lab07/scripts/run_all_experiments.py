"""
Master orchestrator script for Lab 07.
Runs all experiments sequentially:
1. Task 1: Train Oversized Baseline CNN (~1M parameters)
2. Task 2 & 3: Global, Layer-wise, and Structured Pruning + Storage & Latency Benchmarks
3. Task 4: Knowledge Distillation (Plain Student vs Distilled Student, ~100k parameters)
4. Task 5: Four-Pillar Efficiency Analysis Synthesis
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab07.scripts.train_baseline import main as run_task1
from lab07.scripts.run_pruning_experiments import main as run_task2_3
from lab07.scripts.run_distillation_experiments import main as run_task4
from lab07.scripts.run_efficiency_analysis import main as run_task5


def main():
    start_time = time.perf_counter()
    print("=================================================================")
    print("   STARTING LAB 07 COMPRESSION EXPERIMENTS (PRUNING & KD)")
    print("=================================================================")

    print("\n[Step 1/4] Running Task 1: Oversized Baseline CNN Training...")
    run_task1()

    print("\n[Step 2/4] Running Task 2 & 3: Pruning Experiments & Latency/Storage Analysis...")
    run_task2_3()

    print("\n[Step 3/4] Running Task 4: Knowledge Distillation Experiments...")
    run_task4()

    print("\n[Step 4/4] Running Task 5: Synthesizing Four Pillars Efficiency Table...")
    run_task5()

    elapsed = time.perf_counter() - start_time
    print("\n=================================================================")
    print(f"   ALL LAB 07 EXPERIMENTS COMPLETED IN {elapsed / 60:.2f} MINUTES")
    print("=================================================================")


if __name__ == "__main__":
    main()
