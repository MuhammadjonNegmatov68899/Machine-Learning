"""
Hardware-safe incremental executor for Task 2B: Layer-wise Unstructured Pruning.
Optimized for fanless MacBook Air M2 (8GB RAM):
- num_workers=0
- batch_size=64
- Immediate incremental checkpointing after each sparsity target
- Explicit garbage collection and MPS cache clearing
"""

import gc
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import psutil
import torch
import torch.nn as nn
import yaml

from lab07.src.benchmark import benchmark_latency, measure_storage_footprint
from lab07.src.dataset import get_fashion_mnist_dataloaders
from lab07.src.models import build_baseline_model
from lab07.src.pruning import (
    apply_layerwise_unstructured_pruning,
    calculate_sparsity_statistics,
)
from lab07.src.training import evaluate, finetune_pruned_model
from lab07.src.utils import get_device, set_seed


def log_mem():
    vm = psutil.virtual_memory()
    return f"RAM Avail: {vm.available / (1024**3):.2f} GB ({vm.percent}% used)"


def main():
    config_path = PROJECT_ROOT / "lab07" / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    device = get_device(config["experiment"].get("device", "auto"))

    results_dir = PROJECT_ROOT / "lab07" / "results"
    ckpt_dir = PROJECT_ROOT / "lab07" / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_ckpt_path = str(ckpt_dir / "baseline_teacher.pt")
    csv_path = results_dir / "layerwise_pruning_results.csv"

    # Load baseline model
    base_model = build_baseline_model(config)
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.to(device)

    # Use num_workers=0 and batch_size=64 for hardware safety on 8GB M2
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir=config["data"].get("preferred_data_dir", "data"),
        batch_size=64,
        test_batch_size=128,
        val_split=0.1,
        seed=seed,
        num_workers=0,
        pin_memory=False,
    )

    base_eval_loss, base_eval_acc = evaluate(base_model, test_loader, nn.CrossEntropyLoss(), device)

    # Check already completed targets
    completed_targets = set()
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        completed_targets = set(existing_df["target_sparsity"].round(4).tolist())
        print(f"Resuming Layer-wise Pruning. Found completed targets: {completed_targets}")
    else:
        # Initialize with baseline target 0.0
        base_cpu_bench = benchmark_latency(
            base_model.to("cpu"), batch_size=64, device=torch.device("cpu"), warmup_runs=10, num_runs=30
        )
        base_storage = measure_storage_footprint(base_model)
        init_row = pd.DataFrame([{
            "target_sparsity": 0.0,
            "actual_sparsity": 0.0,
            "pre_finetune_test_acc": base_eval_acc,
            "post_finetune_test_acc": base_eval_acc,
            "pre_finetune_test_loss": base_eval_loss,
            "post_finetune_test_loss": base_eval_loss,
            "finetune_time_sec": 0.0,
            "total_params": base_model.count_parameters(),
            "nonzero_params": base_model.count_parameters(),
            "cpu_median_latency_ms": base_cpu_bench["median_latency_ms"],
            "cpu_mean_latency_ms": base_cpu_bench["mean_latency_ms"],
            "cpu_throughput": base_cpu_bench["throughput_samples_sec"],
            **base_storage,
        }])
        init_row.to_csv(csv_path, index=False)
        completed_targets.add(0.0)
        print("Initialized layerwise_pruning_results.csv with 0.0% baseline.")
        base_model.to(device)

    targets = [0.25, 0.50, 0.75, 0.90, 0.95]
    for s in targets:
        if round(s, 4) in completed_targets:
            print(f"Target {s*100:.1f}% already completed. Skipping.")
            continue

        print(f"\n>>> Running Layer-wise Sparsity: {s*100:.1f}% | {log_mem()} <<<")
        pruned_m, masks = apply_layerwise_unstructured_pruning(base_model, s)
        stats = calculate_sparsity_statistics(pruned_m)

        res = finetune_pruned_model(
            model=pruned_m,
            masks=masks,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=3,
            lr=0.0005,
            device=device,
        )

        cpu_bench = benchmark_latency(
            pruned_m.to("cpu"), batch_size=64, device=torch.device("cpu"), warmup_runs=10, num_runs=25
        )
        storage = measure_storage_footprint(pruned_m)

        row = pd.DataFrame([{
            "target_sparsity": s,
            "actual_sparsity": stats["overall_sparsity_ratio"],
            "pre_finetune_test_acc": res["pre_finetune_test_acc"],
            "post_finetune_test_acc": res["post_finetune_test_acc"],
            "pre_finetune_test_loss": res["pre_finetune_test_loss"],
            "post_finetune_test_loss": res["post_finetune_test_loss"],
            "finetune_time_sec": res["finetune_time_sec"],
            "total_params": stats["total_prunable_params"],
            "nonzero_params": stats["total_prunable_params"] - stats["total_zeros"],
            "cpu_median_latency_ms": cpu_bench["median_latency_ms"],
            "cpu_mean_latency_ms": cpu_bench["mean_latency_ms"],
            "cpu_throughput": cpu_bench["throughput_samples_sec"],
            **storage,
        }])

        row.to_csv(csv_path, mode="a", header=False, index=False)
        print(
            f"Saved target {s*100:.1f}%: Pre-Acc={res['pre_finetune_test_acc']*100:.2f}% -> "
            f"Post-Acc={res['post_finetune_test_acc']*100:.2f}%, CPU Latency={cpu_bench['median_latency_ms']:.2f} ms"
        )

        # Cleanup memory immediately
        del pruned_m, masks, res, cpu_bench, storage, row
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        # Brief pause to allow thermal dissipation on fanless M2
        time.sleep(2)

    print("\n=== Task 2B Layer-wise Pruning Completed Successfully! ===")


if __name__ == "__main__":
    main()
