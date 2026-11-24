"""
Task 2 & Task 3: Magnitude & Structured Pruning + Latency & Storage Benchmarking.
Executes:
1. Global unstructured magnitude pruning at [0, 0.25, 0.5, 0.75, 0.9, 0.95] + 3-epoch fine-tuning.
2. Layer-wise unstructured magnitude pruning at [0, 0.25, 0.5, 0.75, 0.9, 0.95] + 3-epoch fine-tuning.
3. Structured filter pruning (L2 norm) at [0.10, 0.25, 0.50] + 3-epoch fine-tuning.
4. Latency and storage footprint analysis (dense vs sparse COO vs compressed).
5. Generation of all publication-quality comparison curves.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import yaml
import copy
import pandas as pd
import torch
import torch.nn as nn

from lab07.src.utils import set_seed, get_device
from lab07.src.dataset import get_fashion_mnist_dataloaders
from lab07.src.models import build_baseline_model
from lab07.src.pruning import (
    apply_global_unstructured_pruning,
    apply_layerwise_unstructured_pruning,
    apply_structured_filter_pruning,
    calculate_sparsity_statistics,
    verify_mask_adherence,
)
from lab07.src.training import finetune_pruned_model, evaluate
from lab07.src.benchmark import (
    benchmark_latency,
    measure_storage_footprint,
    profile_full_model_metrics,
)
from lab07.src.visualization import (
    plot_global_pruning,
    plot_layerwise_pruning,
    plot_structured_pruning,
    plot_combined_pruning_comparison,
    plot_storage_and_latency_analysis,
)


def main():
    config_path = PROJECT_ROOT / "lab07" / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    device = get_device(config["experiment"].get("device", "auto"))

    ckpt_dir = PROJECT_ROOT / "lab07" / "checkpoints"
    results_dir = PROJECT_ROOT / "lab07" / "results"
    plots_dir = PROJECT_ROOT / "lab07" / "plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    baseline_ckpt_path = str(ckpt_dir / "baseline_teacher.pt")
    if not os.path.exists(baseline_ckpt_path):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {baseline_ckpt_path}. Run train_baseline.py first!"
        )

    # Load baseline model
    base_model = build_baseline_model(config)
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.to(device)

    # Data loaders
    data_cfg = config["data"]
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir=data_cfg.get("preferred_data_dir", "data"),
        batch_size=data_cfg.get("batch_size", 128),
        test_batch_size=data_cfg.get("test_batch_size", 256),
        val_split=data_cfg.get("val_split", 0.1),
        seed=seed,
        num_workers=data_cfg.get("num_workers", 2),
    )

    prune_cfg = config["pruning"]
    global_targets = prune_cfg.get("global_sparsity_targets", [0.0, 0.25, 0.50, 0.75, 0.90, 0.95])
    layerwise_targets = prune_cfg.get("layerwise_sparsity_targets", [0.0, 0.25, 0.50, 0.75, 0.90, 0.95])
    structured_ratios = prune_cfg.get("structured_pruning_ratios", [0.10, 0.25, 0.50])
    finetune_epochs = prune_cfg.get("finetune_epochs", 3)
    finetune_lr = float(prune_cfg.get("finetune_lr", 0.0005))

    # ==========================================
    # 1. Global Unstructured Magnitude Pruning
    # ==========================================
    global_csv_path = results_dir / "global_pruning_results.csv"
    base_eval_loss, base_eval_acc = evaluate(base_model, test_loader, nn.CrossEntropyLoss(), device)

    if global_csv_path.exists():
        print(f"\n--- Found completed Global Pruning results at {global_csv_path}. Loading... ---")
        global_df = pd.read_csv(global_csv_path)
    else:
        print("\n--- Running Global Unstructured Magnitude Pruning ---")
        global_results = []
        for s in global_targets:
            print(f"Target Sparsity: {s * 100:.1f}%")
            pruned_m, masks = apply_global_unstructured_pruning(base_model, s)
            stats = calculate_sparsity_statistics(pruned_m)

            if s == 0.0:
                res = {
                    "pre_finetune_test_acc": base_eval_acc,
                    "post_finetune_test_acc": base_eval_acc,
                    "pre_finetune_test_loss": base_eval_loss,
                    "post_finetune_test_loss": base_eval_loss,
                    "finetune_time_sec": 0.0,
                }
            else:
                res = finetune_pruned_model(
                    model=pruned_m,
                    masks=masks,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    epochs=finetune_epochs,
                    lr=finetune_lr,
                    device=device,
                )

            # Profile CPU latency and storage
            cpu_bench = benchmark_latency(
                pruned_m.to("cpu"), batch_size=64, device=torch.device("cpu"), warmup_runs=10, num_runs=30
            )
            storage = measure_storage_footprint(pruned_m)

            record = {
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
            }
            global_results.append(record)
            print(
                f"  Pre-FT Acc: {res['pre_finetune_test_acc']*100:.2f}% -> "
                f"Post-FT Acc: {res['post_finetune_test_acc']*100:.2f}%, "
                f"CPU Latency: {cpu_bench['median_latency_ms']:.2f} ms"
            )

        global_df = pd.DataFrame(global_results)
        global_df.to_csv(global_csv_path, index=False)

    # ==========================================
    # 2. Layer-wise Unstructured Magnitude Pruning
    # ==========================================
    layerwise_csv_path = results_dir / "layerwise_pruning_results.csv"
    layerwise_results = []
    completed_layerwise = set()

    if layerwise_csv_path.exists():
        existing_ldf = pd.read_csv(layerwise_csv_path)
        completed_layerwise = set(existing_ldf["target_sparsity"].round(4).tolist())
        layerwise_results = existing_ldf.to_dict("records")
        print(f"\n--- Found completed Layer-wise targets: {completed_layerwise} at {layerwise_csv_path}. Resuming... ---")
    else:
        print("\n--- Running Layer-wise Unstructured Magnitude Pruning ---")

    for s in layerwise_targets:
        if round(s, 4) in completed_layerwise:
            print(f"Layer-wise target {s * 100:.1f}% already completed. Skipping.")
            continue

        print(f"Layer-wise Target Sparsity: {s * 100:.1f}%")
        pruned_m, masks = apply_layerwise_unstructured_pruning(base_model, s)
        stats = calculate_sparsity_statistics(pruned_m)

        if s == 0.0:
            res = {
                "pre_finetune_test_acc": base_eval_acc,
                "post_finetune_test_acc": base_eval_acc,
                "pre_finetune_test_loss": base_eval_loss,
                "post_finetune_test_loss": base_eval_loss,
                "finetune_time_sec": 0.0,
            }
        else:
            res = finetune_pruned_model(
                model=pruned_m,
                masks=masks,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=finetune_epochs,
                lr=finetune_lr,
                device=device,
            )

        cpu_bench = benchmark_latency(
            pruned_m.to("cpu"), batch_size=64, device=torch.device("cpu"), warmup_runs=10, num_runs=25
        )
        storage = measure_storage_footprint(pruned_m)

        record = {
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
        }
        layerwise_results.append(record)
        pd.DataFrame(layerwise_results).to_csv(layerwise_csv_path, index=False)
        print(
            f"  Pre-FT Acc: {res['pre_finetune_test_acc']*100:.2f}% -> "
            f"Post-FT Acc: {res['post_finetune_test_acc']*100:.2f}%, "
            f"CPU Latency: {cpu_bench['median_latency_ms']:.2f} ms"
        )

    layerwise_df = pd.DataFrame(layerwise_results)
    layerwise_df.to_csv(layerwise_csv_path, index=False)

    # ==========================================
    # 3. Structured Filter Pruning (L2 Norm)
    # ==========================================
    structured_csv_path = results_dir / "structured_pruning_results.csv"
    structured_results = []
    completed_structured = set()

    if structured_csv_path.exists():
        existing_sdf = pd.read_csv(structured_csv_path)
        completed_structured = set(existing_sdf["pruning_ratio"].round(4).tolist())
        structured_results = existing_sdf.to_dict("records")
        print(f"\n--- Found completed Structured targets: {completed_structured} at {structured_csv_path}. Resuming... ---")
    else:
        print("\n--- Running Structured Filter Pruning ---")
        base_cpu_bench = benchmark_latency(
            base_model.to("cpu"), batch_size=64, device=torch.device("cpu"), warmup_runs=10, num_runs=25
        )
        base_storage = measure_storage_footprint(base_model)
        structured_results.append({
            "pruning_ratio": 0.0,
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
        })
        pd.DataFrame(structured_results).to_csv(structured_csv_path, index=False)
        completed_structured.add(0.0)

    for r in structured_ratios:
        if round(r, 4) in completed_structured:
            print(f"Structured ratio {r * 100:.1f}% already completed. Skipping.")
            continue

        print(f"Filter Pruning Ratio: {r * 100:.1f}%")
        struct_m = apply_structured_filter_pruning(base_model, r)
        p_count = struct_m.count_parameters()

        res = finetune_pruned_model(
            model=struct_m,
            masks=None,  # Physically smaller dense model
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=finetune_epochs,
            lr=finetune_lr,
            device=device,
        )

        cpu_bench = benchmark_latency(
            struct_m.to("cpu"), batch_size=64, device=torch.device("cpu"), warmup_runs=10, num_runs=25
        )
        storage = measure_storage_footprint(struct_m)

        record = {
            "pruning_ratio": r,
            "pre_finetune_test_acc": res["pre_finetune_test_acc"],
            "post_finetune_test_acc": res["post_finetune_test_acc"],
            "pre_finetune_test_loss": res["pre_finetune_test_loss"],
            "post_finetune_test_loss": res["post_finetune_test_loss"],
            "finetune_time_sec": res["finetune_time_sec"],
            "total_params": p_count,
            "nonzero_params": p_count,
            "cpu_median_latency_ms": cpu_bench["median_latency_ms"],
            "cpu_mean_latency_ms": cpu_bench["mean_latency_ms"],
            "cpu_throughput": cpu_bench["throughput_samples_sec"],
            **storage,
        }
        structured_results.append(record)
        pd.DataFrame(structured_results).to_csv(structured_csv_path, index=False)
        print(
            f"  Params: {p_count:,} ({p_count / base_model.count_parameters() * 100:.1f}%), "
            f"Pre-FT Acc: {res['pre_finetune_test_acc']*100:.2f}% -> "
            f"Post-FT Acc: {res['post_finetune_test_acc']*100:.2f}%, "
            f"CPU Latency: {cpu_bench['median_latency_ms']:.2f} ms"
        )

    structured_df = pd.DataFrame(structured_results)
    structured_df.to_csv(structured_csv_path, index=False)

    # ==========================================
    # 4. Latency and Storage Analysis
    # ==========================================
    print("\n--- Assembling Latency & Storage Analysis ---")
    storage_df = pd.DataFrame({
        "sparsity": global_df["target_sparsity"],
        "dense_size_kb": global_df["dense_size_kb"],
        "compressed_size_kb": global_df["compressed_size_kb"],
        "sparse_coo_size_kb": global_df["sparse_coo_size_kb"],
    })
    storage_df.to_csv(results_dir / "storage_analysis.csv", index=False)

    # Match structured rows for latency comparison
    base_params = base_model.count_parameters()
    latency_records = []
    for s, unstruct_lat in zip(global_df["target_sparsity"], global_df["cpu_median_latency_ms"]):
        # Find closest structured point if available
        latency_records.append({
            "sparsity": s,
            "unstructured_cpu_ms": unstruct_lat,
        })
    latency_df = pd.DataFrame(latency_records)

    # Add structured latency series
    struct_lat_series = []
    struct_param_red = []
    for idx, row in structured_df.iterrows():
        red = 1.0 - (row["total_params"] / base_params)
        struct_param_red.append(red)
        struct_lat_series.append(row["cpu_median_latency_ms"])

    # Interpolate/pad for plotting
    latency_df["structured_param_reduction"] = pd.Series(struct_param_red)
    latency_df["structured_cpu_ms"] = pd.Series(struct_lat_series)
    latency_df.to_csv(results_dir / "latency_analysis.csv", index=False)

    # ==========================================
    # 5. Visualizations
    # ==========================================
    print("Generating publication-quality figures in lab07/plots/...")
    plot_global_pruning(global_df, str(plots_dir / "global_pruning_curve.png"))
    plot_layerwise_pruning(layerwise_df, str(plots_dir / "layerwise_pruning_curve.png"))
    plot_structured_pruning(structured_df[structured_df["pruning_ratio"] > 0], str(plots_dir / "structured_pruning_curve.png"))
    plot_combined_pruning_comparison(global_df, layerwise_df, structured_df, str(plots_dir / "pruning_methods_comparison.png"))
    plot_storage_and_latency_analysis(storage_df, latency_df.dropna(), str(plots_dir / "storage_and_latency_tradeoffs.png"))

    print("Task 2 and Task 3 experiments completed successfully!")


if __name__ == "__main__":
    main()
