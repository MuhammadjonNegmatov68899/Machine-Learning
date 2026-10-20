#!/usr/bin/env python3
"""
Main driver script for Machine Learning Lab 3: Architectures and Their Costs.
Executes dataset loading, parameter counting, FLOPs profiling, latency benchmarking,
model training, test evaluation, and plot/CSV generation.
"""

import argparse
import os
import sys
import pandas as pd
import torch

from src.dataset import get_fashion_mnist_dataloaders
from src.models import get_all_models
from src.compute_cost import (
    hand_calculate_layer_costs,
    hand_calculate_full_model_costs,
    profile_model,
)
from src.latency import benchmark_model_latencies
from src.train import train_model, set_seed
from src.plot_results import (
    plot_accuracy_vs_parameters,
    plot_accuracy_vs_latency,
    plot_training_history,
)


def run_experiments(
    arch_mode: str = "specified",
    device_str: str = "auto",
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    results_dir: str = "results",
) -> None:
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # Determine device
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"================================================================")
    print(f" Machine Learning Lab 3: Architectures and Their Costs")
    print(f" Execution Device : {device}")
    print(f" Architecture Mode: {arch_mode}")
    print(f" Random Seed      : {seed}")
    print(f" Training Epochs  : {epochs} | LR: {lr} | Batch Size: {batch_size}")
    print(f"================================================================\n")

    # 1. Dataset loading
    print("[1/5] Loading Fashion-MNIST dataset...")
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir="./data",
        batch_size=batch_size,
        seed=seed,
    )
    print(f"      Training set   : 55,000 samples ({len(train_loader)} batches of {batch_size})")
    print(f"      Validation set :  5,000 samples ({len(val_loader)} batches)")
    print(f"      Test set       : 10,000 samples ({len(test_loader)} batches)\n")

    # 2. Hand calculations for representative layers
    print("[2/5] Performing Hand Calculations for Representative Layers...")
    layer_costs = hand_calculate_layer_costs()
    layer_records = []
    for layer_name, data in layer_costs.items():
        layer_records.append({
            "Layer": layer_name,
            "Parameters": data["params"],
            "Hand_MACs": data["macs"],
            "Hand_FLOPs": data["flops"],
            "MAC_Formula": data["formula_macs"],
            "FLOP_Formula": data["formula_flops"],
        })
    df_layers = pd.DataFrame(layer_records)
    layer_csv_path = os.path.join(results_dir, "layer_hand_costs.csv")
    df_layers.to_csv(layer_csv_path, index=False)
    print(f"      Saved representative layer costs to {layer_csv_path}")
    print(df_layers[["Layer", "Parameters", "Hand_MACs", "Hand_FLOPs"]].to_string(index=False))
    print()

    # Determine modes to evaluate
    modes_to_run = ["specified"] if arch_mode == "specified" else (["calibrated"] if arch_mode == "calibrated" else ["specified", "calibrated"])

    all_summary_data = []
    all_latency_records = []
    all_profiler_records = []
    all_training_histories = {}

    for mode in modes_to_run:
        print(f"================================================================")
        print(f" Processing Architectures in Mode: '{mode.upper()}'")
        print(f"================================================================")
        models_dict = get_all_models(mode=mode)

        for arch_name, model in models_dict.items():
            run_key = f"{arch_name}_{mode}"
            print(f"\n>>> Analyzing Architecture: {arch_name} ({mode})")

            # 3. Model Profiling & Parameter Counting
            hand_totals = hand_calculate_full_model_costs(arch_name, mode=mode)
            prof = profile_model(model, input_size=(1, 1, 28, 28))

            print(f"    - Parameters (Hand-calc)   : {hand_totals['params']:,}")
            print(f"    - Parameters (torchinfo)   : {prof['torchinfo_params']:,}")
            print(f"    - Parameters (thop raw)    : {prof['thop_raw_params'] if prof['thop_raw_params'] is not None else 'N/A'}")
            print(f"    - Hand MACs (1 sample)     : {hand_totals['macs']:,}")
            print(f"    - Hand FLOPs (2*MACs)      : {hand_totals['flops']:,}")
            print(f"    - torchinfo Mult-Adds      : {prof['torchinfo_mult_adds']:,}")
            print(f"    - thop Native MACs (raw)   : {prof['thop_raw_macs']:,} (skips transformer blocks if unhooked)")
            print(f"    - thop Custom-Hooked MACs  : {prof['thop_custom_macs']:,}")

            all_profiler_records.append({
                "model": arch_name,
                "mode": mode,
                "hand_params": hand_totals["params"],
                "torchinfo_params": prof["torchinfo_params"],
                "thop_raw_params": prof["thop_raw_params"],
                "hand_macs": hand_totals["macs"],
                "hand_flops": hand_totals["flops"],
                "torchinfo_mult_adds": prof["torchinfo_mult_adds"],
                "thop_raw_macs": prof["thop_raw_macs"],
                "thop_custom_macs": prof["thop_custom_macs"],
                "effective_macs": prof["effective_macs"],
                "effective_flops": prof["effective_flops"],
            })

            # 4. Latency Benchmarking (Batch 1: 100 runs, Batch 128: 20 runs)
            print(f"    - Benchmarking inference latency on {device}...")
            # Also benchmark on CPU to report hardware-agnostic baseline
            lat_dev = benchmark_model_latencies(model, device=device)
            lat_cpu = benchmark_model_latencies(model, device=torch.device("cpu"))
            
            lat_b1_med = lat_dev["batch_1"]["median_ms"]
            lat_b128_med = lat_dev["batch_128"]["median_ms"]
            cpu_b1_med = lat_cpu["batch_1"]["median_ms"]
            cpu_b128_med = lat_cpu["batch_128"]["median_ms"]

            print(f"      [{device}]  Batch   1 (100 runs) median: {lat_b1_med:.4f} ms")
            print(f"      [{device}]  Batch 128 ( 20 runs) median: {lat_b128_med:.4f} ms (per sample: {lat_dev['batch_128']['per_sample_ms']:.4f} ms)")
            print(f"      [CPU]  Batch   1 (100 runs) median: {cpu_b1_med:.4f} ms")
            print(f"      [CPU]  Batch 128 ( 20 runs) median: {cpu_b128_med:.4f} ms (per sample: {lat_cpu['batch_128']['per_sample_ms']:.4f} ms)")

            all_latency_records.append({
                "model": arch_name,
                "mode": mode,
                "device": str(device),
                "b1_median_ms": lat_b1_med,
                "b1_mean_ms": lat_dev["batch_1"]["mean_ms"],
                "b1_std_ms": lat_dev["batch_1"]["std_ms"],
                "b128_median_ms": lat_b128_med,
                "b128_mean_ms": lat_dev["batch_128"]["mean_ms"],
                "b128_std_ms": lat_dev["batch_128"]["std_ms"],
                "cpu_b1_median_ms": cpu_b1_med,
                "cpu_b128_median_ms": cpu_b128_med,
            })

            # 5. Training
            train_res = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=seed,
            )

            test_acc = train_res["test_acc"]
            test_loss = train_res["test_loss"]
            all_training_histories[run_key] = train_res["history"]

            all_summary_data.append({
                "model": arch_name,
                "mode": mode,
                "params": prof["torchinfo_params"],
                "hand_macs": hand_totals["macs"],
                "hand_flops": hand_totals["flops"],
                "thop_raw_macs": prof["thop_raw_macs"],
                "verified_macs": prof["effective_macs"],
                "verified_flops": prof["effective_flops"],
                "latency_b1_ms": lat_b1_med,
                "latency_b128_ms": lat_b128_med,
                "cpu_latency_b1_ms": cpu_b1_med,
                "cpu_latency_b128_ms": cpu_b128_med,
                "test_loss": test_loss,
                "test_acc": test_acc,
            })

    # Save CSVs
    df_summary = pd.DataFrame(all_summary_data)
    summary_csv_path = os.path.join(results_dir, "summary_metrics.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    df_lat = pd.DataFrame(all_latency_records)
    lat_csv_path = os.path.join(results_dir, "latency_results.csv")
    df_lat.to_csv(lat_csv_path, index=False)

    df_prof = pd.DataFrame(all_profiler_records)
    prof_csv_path = os.path.join(results_dir, "profiler_comparison.csv")
    df_prof.to_csv(prof_csv_path, index=False)

    # Flatten training history for CSV
    history_rows = []
    for run_key, hist in all_training_histories.items():
        arch, mode = run_key.split("_")
        for i in range(len(hist["epoch"])):
            history_rows.append({
                "model": arch,
                "mode": mode,
                "epoch": hist["epoch"][i],
                "train_loss": hist["train_loss"][i],
                "val_loss": hist["val_loss"][i],
                "val_acc": hist["val_acc"][i],
                "epoch_time_sec": hist["epoch_time_sec"][i],
            })
    df_history = pd.DataFrame(history_rows)
    history_csv_path = os.path.join(results_dir, "training_history.csv")
    df_history.to_csv(history_csv_path, index=False)

    # Generate Figures
    print("\n[5/5] Generating Visualization Plots...")
    plot_accuracy_vs_parameters(
        all_summary_data,
        output_path=os.path.join(results_dir, "accuracy_vs_params.png"),
    )
    plot_accuracy_vs_latency(
        all_summary_data,
        output_path=os.path.join(results_dir, "accuracy_vs_latency.png"),
    )
    plot_training_history(
        all_training_histories,
        output_path=os.path.join(results_dir, "training_curves.png"),
    )

    print("\n================================================================")
    print(" EXPERIMENT COMPLETE — SUMMARY OF RESULTS")
    print("================================================================")
    print(df_summary[["model", "mode", "params", "hand_flops", "verified_macs", "verified_flops", "thop_raw_macs", "latency_b1_ms", "latency_b128_ms", "test_acc"]].to_string(index=False))
    print(f"\nAll artifacts successfully saved under '{results_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lab 3 Experiments")
    parser.add_argument("--arch-mode", type=str, default="all", choices=["specified", "calibrated", "all"],
                        help="Architecture mode: 'specified' (manual spec), 'calibrated' (~100k-150k target), or 'all'")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Device to use for training and latency benchmarking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSVs and plots")

    args = parser.parse_args()
    run_experiments(
        arch_mode=args.arch_mode,
        device_str=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        results_dir=args.results_dir,
    )
