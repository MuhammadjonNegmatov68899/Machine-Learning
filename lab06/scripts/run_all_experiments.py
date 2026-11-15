#!/usr/bin/env python3
"""
Master Orchestrator for Lab 6:
Executes Tasks 1 through 7, generates all CSV tables, profiler summaries,
and publication-ready visualization plots.

Usage:
    python lab06/scripts/run_all_experiments.py [--tasks all] [--device auto]
"""

import argparse
import sys
import os
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from lab06.src.pipeline import train_pipeline
from lab06.src.profiler import profile_training_epoch
from lab06.src.studies import (
    run_workers_study,
    run_batch_size_study,
    run_optimizer_scheduler_study,
    run_gradient_accumulation_study,
    run_checkpoint_resume_study,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Master runner for Lab 6 experiments.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated list of tasks to run (e.g. '1,2,3' or 'all'). Default: 'all'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lab06/configs/train_config.yaml",
        help="Path to YAML training configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device ('auto', 'mps', 'cpu', 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks_input = args.tasks.lower().strip()
    if tasks_input == "all":
        tasks_to_run = [1, 2, 3, 4, 5, 6, 7]
    else:
        tasks_to_run = [int(t.strip()) for t in tasks_input.split(",") if t.strip()]

    print("=================================================================")
    print("        Machine Learning Coursework: Lab 6 Full Benchmark        ")
    print(f" Tasks to Execute: {tasks_to_run}")
    print(f" Target Device:    {args.device}")
    print(f" Base Config:      {args.config}")
    print("=================================================================")

    start_total_time = time.time()
    results_summary = {}

    # -------------------------------------------------------------------------
    # Task 1: YAML-Driven Pipeline with MLflow & Checkpointing
    # -------------------------------------------------------------------------
    if 1 in tasks_to_run:
        print("\n" + "=" * 60)
        print(">>> TASK 1: YAML-Driven Training Pipeline <<<")
        print("=" * 60)
        t0 = time.time()
        res1 = train_pipeline(args.config, override_device=args.device)
        elapsed1 = time.time() - t0
        results_summary["Task 1"] = {
            "status": "PASSED",
            "time_sec": elapsed1,
            "test_acc": f"{res1['test_acc']:.2f}%",
            "artifact": res1["best_checkpoint_path"],
        }

    # -------------------------------------------------------------------------
    # Task 2: PyTorch Profiler Breakdown
    # -------------------------------------------------------------------------
    if 2 in tasks_to_run:
        print("\n" + "=" * 60)
        print(">>> TASK 2: PyTorch Profiling (1 Epoch Breakdown) <<<")
        print("=" * 60)
        t0 = time.time()
        res2 = profile_training_epoch(
            data_dir="lab03/data",
            batch_size=128,
            num_workers=0,
            device_str=args.device,
            export_trace=True,
            output_dir="lab06/results",
            plot_dir="lab06/plots",
        )
        elapsed2 = time.time() - t0
        shares = res2["shares"]
        results_summary["Task 2"] = {
            "status": "PASSED",
            "time_sec": elapsed2,
            "data_loading_pct": f"{shares['data_loading']:.1f}%",
            "forward_pct": f"{shares['forward_pass']:.1f}%",
            "backward_pct": f"{shares['backward_pass']:.1f}%",
            "optimizer_pct": f"{shares['optimizer_step']:.1f}%",
        }

    # -------------------------------------------------------------------------
    # Task 3: DataLoader Workers Study
    # -------------------------------------------------------------------------
    if 3 in tasks_to_run:
        print("\n" + "=" * 60)
        print(">>> TASK 3: DataLoader Workers Study [0, 1, 2, 4, 8] <<<")
        print("=" * 60)
        t0 = time.time()
        res3 = run_workers_study(
            data_dir="lab03/data",
            workers_list=[0, 1, 2, 4, 8],
            batch_size=128,
            epochs_per_setting=1,
            device_str=args.device,
            output_dir="lab06/results",
            plot_dir="lab06/plots",
        )
        elapsed3 = time.time() - t0
        best_worker = res3.loc[res3["throughput_samples_per_sec"].idxmax()]
        results_summary["Task 3"] = {
            "status": "PASSED",
            "time_sec": elapsed3,
            "peak_throughput": f"{best_worker['throughput_samples_per_sec']:.1f} smp/s (workers={int(best_worker['num_workers'])})",
        }

    # -------------------------------------------------------------------------
    # Task 4: Batch Size Study
    # -------------------------------------------------------------------------
    if 4 in tasks_to_run:
        print("\n" + "=" * 60)
        print(">>> TASK 4: Batch Size Study [32, 64, 128, 256, 512] <<<")
        print("=" * 60)
        t0 = time.time()
        res4 = run_batch_size_study(
            data_dir="lab03/data",
            batch_sizes=[32, 64, 128, 256, 512],
            epochs=5,
            device_str=args.device,
            output_dir="lab06/results",
            plot_dir="lab06/plots",
        )
        elapsed4 = time.time() - t0
        best_acc_row = res4.loc[res4["final_test_acc"].idxmax()]
        results_summary["Task 4"] = {
            "status": "PASSED",
            "time_sec": elapsed4,
            "highest_acc": f"{best_acc_row['final_test_acc']:.2f}% (bs={int(best_acc_row['batch_size'])})",
        }

    # -------------------------------------------------------------------------
    # Task 5: Optimizers and Schedulers Study
    # -------------------------------------------------------------------------
    if 5 in tasks_to_run:
        print("\n" + "=" * 60)
        print(">>> TASK 5: Optimizers & Schedulers Study (6 configs x 10 epochs) <<<")
        print("=" * 60)
        t0 = time.time()
        sum5, _ = run_optimizer_scheduler_study(
            data_dir="lab03/data",
            epochs=10,
            batch_size=128,
            device_str=args.device,
            output_dir="lab06/results",
            plot_dir="lab06/plots",
        )
        elapsed5 = time.time() - t0
        best_cfg_row = sum5.loc[sum5["final_test_acc"].idxmax()]
        results_summary["Task 5"] = {
            "status": "PASSED",
            "time_sec": elapsed5,
            "champion_config": f"{best_cfg_row['config']} ({best_cfg_row['final_test_acc']:.2f}%)",
        }

    # -------------------------------------------------------------------------
    # Task 6: Gradient Accumulation Study
    # -------------------------------------------------------------------------
    if 6 in tasks_to_run:
        print("\n" + "=" * 60)
        print(">>> TASK 6: Gradient Accumulation (BS 256 vs. Physical 64 x 4) <<<")
        print("=" * 60)
        t0 = time.time()
        res6 = run_gradient_accumulation_study(
            data_dir="lab03/data",
            epochs=5,
            device_str=args.device,
            output_dir="lab06/results",
            plot_dir="lab06/plots",
        )
        elapsed6 = time.time() - t0
        results_summary["Task 6"] = {
            "status": "PASSED",
            "time_sec": elapsed6,
            "std_acc": f"{res6.iloc[0]['final_test_acc']:.2f}%",
            "accum_acc": f"{res6.iloc[1]['final_test_acc']:.2f}%",
        }

    # -------------------------------------------------------------------------
    # Task 7: Checkpoint Interruption and Resumption
    # -------------------------------------------------------------------------
    if 7 in tasks_to_run:
        print("\n" + "=" * 60)
        print(">>> TASK 7: Checkpoint Interruption & Resumption (Epochs 1-3 -> 4-6) <<<")
        print("=" * 60)
        t0 = time.time()
        res7 = run_checkpoint_resume_study(
            data_dir="lab03/data",
            total_epochs=6,
            interrupt_epoch=3,
            batch_size=128,
            device_str=args.device,
            output_dir="lab06/results",
            checkpoint_dir="lab06/checkpoints",
            plot_dir="lab06/plots",
        )
        elapsed7 = time.time() - t0
        max_diff = res7["val_acc_diff"].max()
        results_summary["Task 7"] = {
            "status": "PASSED",
            "time_sec": elapsed7,
            "max_val_acc_diff": f"{max_diff:.4f}%",
        }

    total_duration = time.time() - start_total_time

    print("\n" + "=" * 70)
    print("                     LAB 6 EXECUTION COMPLETE                     ")
    print(f" Total Duration: {total_duration:.2f}s ({total_duration / 60.0:.2f} min)")
    print("=" * 70)
    for task_name, info in results_summary.items():
        print(f"  {task_name:<10}: {info['status']} ({info['time_sec']:.1f}s) -> {info}")
    print("=" * 70)


if __name__ == "__main__":
    main()
