"""
Empirical Studies for Lab 6:
- Task 3: DataLoader Workers Study (num_workers: [0, 1, 2, 4, 8])
- Task 4: Batch Size Study (batch_size: [32, 64, 128, 256, 512])
- Task 5: Optimizers and LR Schedulers (SGD+momentum, Adam x StepLR, Cosine, Plateau)
- Task 6: Gradient Accumulation (Batch 256 vs Physical 64 with Accumulation 4)
- Task 7: Checkpoint Interruption and Resumption (Epoch 1-3, save, resume 4-6)
"""

from typing import Dict, Any, List, Tuple
import os
import time
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .models import CNN
from .dataset import get_fashion_mnist_dataloaders
from .pipeline import evaluate, save_checkpoint, load_checkpoint
from .utils import set_seed, get_device, get_memory_info, setup_plot_style


# ==============================================================================
# Task 3: DataLoader Workers Study
# ==============================================================================

def run_workers_study(
    data_dir: str = "lab03/data",
    workers_list: List[int] = [0, 1, 2, 4, 8],
    batch_size: int = 128,
    epochs_per_setting: int = 1,
    device_str: str = "auto",
    seed: int = 42,
    output_dir: str = "lab06/results",
    plot_dir: str = "lab06/plots",
) -> pd.DataFrame:
    """
    Evaluates training throughput and epoch duration across varying num_workers settings.
    Identifies system bottlenecks (I/O, IPC overhead, or compute saturation).
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    device = get_device(device_str)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=======================================================")
    print(f" Task 3: DataLoader Workers Study {workers_list}")
    print(f" Device: {device} | Batch Size: {batch_size}")
    print(f"=======================================================")

    results = []

    for num_workers in workers_list:
        set_seed(seed)
        train_loader, _, _ = get_fashion_mnist_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
        )

        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        total_samples = 0
        total_time = 0.0

        for ep in range(epochs_per_setting):
            t_start = time.perf_counter()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_samples += inputs.size(0)

            if device.type == "mps":
                torch.mps.synchronize()
            total_time += (time.perf_counter() - t_start)

        avg_epoch_time = total_time / epochs_per_setting
        throughput = total_samples / total_time

        print(
            f"num_workers: {num_workers:2d} | "
            f"Epoch Time: {avg_epoch_time:6.2f}s | "
            f"Throughput: {throughput:7.1f} samples/s"
        )

        results.append({
            "num_workers": num_workers,
            "epoch_time_sec": avg_epoch_time,
            "throughput_samples_per_sec": throughput,
            "total_samples": total_samples,
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "task3_workers_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Task 3] Saved results to {csv_path}")

    # Plot
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Throughput plot
    ax1.plot(df["num_workers"], df["throughput_samples_per_sec"], marker="o", color="#1f77b4", linewidth=2.2, markersize=8)
    for _, row in df.iterrows():
        ax1.annotate(
            f"{row['throughput_samples_per_sec']:.1f}",
            (row["num_workers"], row["throughput_samples_per_sec"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    ax1.set_xlabel("Number of DataLoader Workers (num_workers)")
    ax1.set_ylabel("Throughput (samples / sec)")
    ax1.set_title("Training Throughput vs. Worker Count")
    ax1.set_xticks(workers_list)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Epoch Time plot
    ax2.plot(df["num_workers"], df["epoch_time_sec"], marker="s", color="#d62728", linewidth=2.2, markersize=8)
    for _, row in df.iterrows():
        ax2.annotate(
            f"{row['epoch_time_sec']:.2f}s",
            (row["num_workers"], row["epoch_time_sec"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    ax2.set_xlabel("Number of DataLoader Workers (num_workers)")
    ax2.set_ylabel("Epoch Time (seconds)")
    ax2.set_title("Training Epoch Duration vs. Worker Count")
    ax2.set_xticks(workers_list)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(plot_dir, "task3_workers_study.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[Task 3] Saved plot to {plot_path}")

    return df


# ==============================================================================
# Task 4: Batch Size Study
# ==============================================================================

def run_batch_size_study(
    data_dir: str = "lab03/data",
    batch_sizes: List[int] = [32, 64, 128, 256, 512],
    epochs: int = 5,
    device_str: str = "auto",
    seed: int = 42,
    output_dir: str = "lab06/results",
    plot_dir: str = "lab06/plots",
) -> pd.DataFrame:
    """
    Evaluates throughput, peak memory usage, and final test accuracy after exactly
    5 epochs across batch sizes [32, 64, 128, 256, 512].
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    device = get_device(device_str)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=======================================================")
    print(f" Task 4: Batch Size Study {batch_sizes}")
    print(f" Device: {device} | Epochs: {epochs}")
    print(f"=======================================================")

    results = []

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} (5 Epochs) ---")
        set_seed(seed)
        train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
            data_dir=data_dir,
            batch_size=bs,
            seed=seed,
            num_workers=0,
        )

        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        total_samples = 0
        total_train_time = 0.0
        peak_rss_mb = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            t_epoch_start = time.perf_counter()

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_samples += inputs.size(0)

                # Monitor memory
                mem = get_memory_info()
                if mem["rss_mb"] > peak_rss_mb:
                    peak_rss_mb = mem["rss_mb"]

            if device.type == "mps":
                torch.mps.synchronize()

            epoch_dur = time.perf_counter() - t_epoch_start
            total_train_time += epoch_dur

        throughput = total_samples / total_train_time if total_train_time > 0 else 0.0

        # Evaluate final test accuracy after exactly 5 epochs
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Batch Size: {bs:3d} | "
            f"Throughput: {throughput:7.1f} smp/s | "
            f"Peak Memory: {peak_rss_mb:6.1f} MB | "
            f"Test Loss: {test_loss:.4f} | "
            f"Final Test Acc: {test_acc:5.2f}%"
        )

        results.append({
            "batch_size": bs,
            "throughput_samples_per_sec": throughput,
            "peak_memory_mb": peak_rss_mb,
            "total_train_time_sec": total_train_time,
            "test_loss": test_loss,
            "final_test_acc": test_acc,
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "task4_batch_size_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Task 4] Saved results to {csv_path}")

    # Plot
    setup_plot_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Throughput vs Batch Size
    ax1.plot(df["batch_size"], df["throughput_samples_per_sec"], marker="o", color="#1f77b4", linewidth=2.2, markersize=8)
    for _, row in df.iterrows():
        ax1.annotate(
            f"{row['throughput_samples_per_sec']:.0f}",
            (row["batch_size"], row["throughput_samples_per_sec"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (samples / sec)")
    ax1.set_title("Throughput vs. Batch Size")
    ax1.set_xticks(batch_sizes)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Peak Memory vs Batch Size
    ax2.plot(df["batch_size"], df["peak_memory_mb"], marker="s", color="#ff7f0e", linewidth=2.2, markersize=8)
    for _, row in df.iterrows():
        ax2.annotate(
            f"{row['peak_memory_mb']:.1f}",
            (row["batch_size"], row["peak_memory_mb"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Peak Memory Usage vs. Batch Size")
    ax2.set_xticks(batch_sizes)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Final Accuracy vs Batch Size
    ax3.plot(df["batch_size"], df["final_test_acc"], marker="^", color="#2ca02c", linewidth=2.2, markersize=8)
    for _, row in df.iterrows():
        ax3.annotate(
            f"{row['final_test_acc']:.2f}%",
            (row["batch_size"], row["final_test_acc"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Final Test Accuracy (%)")
    ax3.set_title("5-Epoch Test Accuracy vs. Batch Size")
    ax3.set_xticks(batch_sizes)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(plot_dir, "task4_batch_size_tradeoffs.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[Task 4] Saved plot to {plot_path}")

    return df


# ==============================================================================
# Task 5: Optimizers and Learning Rate Schedulers Study
# ==============================================================================

def run_optimizer_scheduler_study(
    data_dir: str = "lab03/data",
    epochs: int = 10,
    batch_size: int = 128,
    device_str: str = "auto",
    seed: int = 42,
    output_dir: str = "lab06/results",
    plot_dir: str = "lab06/plots",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares 2 optimizers (SGD+momentum, Adam) across 3 schedulers
    (StepLR, CosineAnnealingLR, ReduceLROnPlateau) under an identical 10-epoch budget.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    device = get_device(device_str)
    criterion = nn.CrossEntropyLoss()

    optimizers_to_test = ["SGD_Momentum", "Adam"]
    schedulers_to_test = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]

    print(f"\n=======================================================")
    print(f" Task 5: Optimizers & LR Schedulers Study")
    print(f" Optimizers: {optimizers_to_test}")
    print(f" Schedulers: {schedulers_to_test}")
    print(f" Budget: {epochs} Epochs each (6 configurations)")
    print(f" Device: {device} | Batch Size: {batch_size}")
    print(f"=======================================================")

    summary_results = []
    history_records = []

    for opt_name in optimizers_to_test:
        for sched_name in schedulers_to_test:
            config_label = f"{opt_name} + {sched_name}"
            print(f"\n>>> Running: {config_label} ({epochs} epochs) <<<")
            set_seed(seed)

            train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                seed=seed,
                num_workers=0,
            )

            model = CNN().to(device)

            # Setup Optimizer
            if opt_name == "SGD_Momentum":
                optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
            else:  # Adam
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            # Setup Scheduler
            if sched_name == "StepLR":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
            elif sched_name == "CosineAnnealingLR":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
            else:  # ReduceLROnPlateau
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=1
                )

            total_train_time = 0.0

            for epoch in range(1, epochs + 1):
                model.train()
                running_loss = 0.0
                running_correct = 0
                total_train = 0
                t_ep_start = time.perf_counter()

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_correct += (outputs.argmax(dim=1) == targets).sum().item()
                    total_train += inputs.size(0)

                if device.type == "mps":
                    torch.mps.synchronize()

                ep_time = time.perf_counter() - t_ep_start
                total_train_time += ep_time

                train_loss = running_loss / total_train
                train_acc = 100.0 * (running_correct / total_train)

                val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                current_lr = optimizer.param_groups[0]["lr"]

                # Step scheduler
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                history_records.append({
                    "config": config_label,
                    "optimizer": opt_name,
                    "scheduler": sched_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "epoch_time_sec": ep_time,
                })

                print(
                    f"Epoch {epoch:2d}/{epochs:2d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:5.2f}% | LR: {current_lr:.6f} | Time: {ep_time:.2f}s"
                )

            # Final Test Evaluation
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            summary_results.append({
                "config": config_label,
                "optimizer": opt_name,
                "scheduler": sched_name,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "final_val_acc": val_acc,
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
                "total_train_time_sec": total_train_time,
                "avg_epoch_time_sec": total_train_time / epochs,
            })

    summary_df = pd.DataFrame(summary_results)
    history_df = pd.DataFrame(history_records)

    summary_csv = os.path.join(output_dir, "task5_optimizers_schedulers_study.csv")
    history_csv = os.path.join(output_dir, "task5_training_histories.csv")
    summary_df.to_csv(summary_csv, index=False)
    history_df.to_csv(history_csv, index=False)

    print(f"\n[Task 5] Saved summary to {summary_csv}")
    print(f"[Task 5] Saved history to {history_csv}")

    # Plot Visualizations
    setup_plot_style()
    colors = {
        "SGD_Momentum + StepLR": "#1f77b4",
        "SGD_Momentum + CosineAnnealingLR": "#aec7e8",
        "SGD_Momentum + ReduceLROnPlateau": "#2ca02c",
        "Adam + StepLR": "#ff7f0e",
        "Adam + CosineAnnealingLR": "#ffbb78",
        "Adam + ReduceLROnPlateau": "#d62728",
    }

    # 1. Loss Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for cfg in summary_df["config"]:
        sub = history_df[history_df["config"] == cfg]
        ax1.plot(sub["epoch"], sub["train_loss"], label=cfg, color=colors.get(cfg), linewidth=2)
        ax2.plot(sub["epoch"], sub["val_loss"], label=cfg, color=colors.get(cfg), linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Convergence (10 Epochs)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Trajectory (10 Epochs)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    loss_plot_path = os.path.join(plot_dir, "task5_loss_curves.png")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)
    plt.close(fig)

    # 2. Validation Accuracy Curves Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in summary_df["config"]:
        sub = history_df[history_df["config"] == cfg]
        ax.plot(sub["epoch"], sub["val_acc"], label=cfg, color=colors.get(cfg), marker="o", markersize=4, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy Across 6 Optimizer-Scheduler Configurations")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    acc_plot_path = os.path.join(plot_dir, "task5_val_accuracy_curves.png")
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=300)
    plt.close(fig)

    # 3. Bar Chart Comparison
    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(14, 5))
    y_pos = np.arange(len(summary_df))
    bars1 = ax_acc.barh(y_pos, summary_df["final_test_acc"], color=[colors.get(c, "#333") for c in summary_df["config"]])
    ax_acc.set_yticks(y_pos)
    ax_acc.set_yticklabels(summary_df["config"], fontsize=9)
    ax_acc.invert_yaxis()
    ax_acc.set_xlabel("Final Test Accuracy (%)")
    ax_acc.set_title("Final Test Accuracy Comparison", fontweight="bold")
    for bar in bars1:
        w = bar.get_width()
        ax_acc.text(w - 3.5, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", ha="right", va="center", color="white", fontweight="bold")

    bars2 = ax_time.barh(y_pos, summary_df["total_train_time_sec"], color=[colors.get(c, "#333") for c in summary_df["config"]])
    ax_time.set_yticks(y_pos)
    ax_time.set_yticklabels(summary_df["config"], fontsize=9)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Total Training Time (seconds)")
    ax_time.set_title("Training Time (10 Epochs)", fontweight="bold")
    for bar in bars2:
        w = bar.get_width()
        ax_time.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}s", ha="left", va="center", fontweight="bold")

    bar_plot_path = os.path.join(plot_dir, "task5_final_metrics_bar.png")
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Task 5] Saved plots to {loss_plot_path}, {acc_plot_path}, {bar_plot_path}")

    return summary_df, history_df


# ==============================================================================
# Task 6: Gradient Accumulation Study
# ==============================================================================

def run_gradient_accumulation_study(
    data_dir: str = "lab03/data",
    epochs: int = 5,
    device_str: str = "auto",
    seed: int = 42,
    output_dir: str = "lab06/results",
    plot_dir: str = "lab06/plots",
) -> pd.DataFrame:
    """
    Compares:
      1. Standard training with physical batch size 256.
      2. Simulated batch size 256 using physical batch size 64 and gradient accumulation steps 4.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    device = get_device(device_str)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=======================================================")
    print(f" Task 6: Gradient Accumulation Study")
    print(f" Target Effective Batch Size: 256")
    print(f" Baseline: Physical Batch 256, Accumulation Steps: 1")
    print(f" Emulated: Physical Batch 64,  Accumulation Steps: 4")
    print(f" Device: {device} | Epochs: {epochs}")
    print(f"=======================================================")

    experiments = [
        {"name": "Standard_BS256", "batch_size": 256, "accum_steps": 1},
        {"name": "Accumulated_BS64_x4", "batch_size": 64, "accum_steps": 4},
    ]

    histories = {}
    summary_results = []

    for exp in experiments:
        exp_name = exp["name"]
        bs = exp["batch_size"]
        accum_steps = exp["accum_steps"]

        print(f"\n>>> Running {exp_name} (Physical BS={bs}, Accum Steps={accum_steps}) <<<")
        set_seed(seed)

        train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
            data_dir=data_dir,
            batch_size=bs,
            seed=seed,
            num_workers=0,
        )

        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        ep_train_losses = []
        ep_val_losses = []
        ep_val_accs = []
        peak_memory_mb = 0.0
        total_time = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            total_samples = 0
            t_ep_start = time.perf_counter()

            optimizer.zero_grad()
            total_batches = len(train_loader)

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Normalize loss by accumulation steps
                loss_scaled = loss / accum_steps
                loss_scaled.backward()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Monitor peak memory
                mem = get_memory_info()
                if mem["rss_mb"] > peak_memory_mb:
                    peak_memory_mb = mem["rss_mb"]

                # Perform optimizer step at accumulation intervals or epoch boundary
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == total_batches:
                    optimizer.step()
                    optimizer.zero_grad()

            if device.type == "mps":
                torch.mps.synchronize()

            ep_time = time.perf_counter() - t_ep_start
            total_time += ep_time

            train_loss = running_loss / total_samples
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            ep_train_losses.append(train_loss)
            ep_val_losses.append(val_loss)
            ep_val_accs.append(val_acc)

            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:5.2f}% | Time: {ep_time:.2f}s"
            )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        histories[exp_name] = {
            "train_loss": ep_train_losses,
            "val_loss": ep_val_losses,
            "val_acc": ep_val_accs,
        }

        summary_results.append({
            "experiment": exp_name,
            "physical_batch_size": bs,
            "accumulation_steps": accum_steps,
            "effective_batch_size": bs * accum_steps,
            "final_train_loss": ep_train_losses[-1],
            "final_val_loss": ep_val_losses[-1],
            "final_val_acc": ep_val_accs[-1],
            "final_test_loss": test_loss,
            "final_test_acc": test_acc,
            "peak_memory_mb": peak_memory_mb,
            "total_time_sec": total_time,
        })

    summary_df = pd.DataFrame(summary_results)
    csv_path = os.path.join(output_dir, "task6_gradient_accumulation.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[Task 6] Saved summary to {csv_path}")

    # Plot Comparison
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = list(range(1, epochs + 1))

    # Loss trajectories
    ax1.plot(epochs_range, histories["Standard_BS256"]["train_loss"], label="Standard BS256 (Train)", color="#1f77b4", linestyle="-", marker="o")
    ax1.plot(epochs_range, histories["Standard_BS256"]["val_loss"], label="Standard BS256 (Val)", color="#1f77b4", linestyle="--", marker="x")
    ax1.plot(epochs_range, histories["Accumulated_BS64_x4"]["train_loss"], label="Accum BS64x4 (Train)", color="#d62728", linestyle="-", marker="s")
    ax1.plot(epochs_range, histories["Accumulated_BS64_x4"]["val_loss"], label="Accum BS64x4 (Val)", color="#d62728", linestyle="--", marker="+")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Convergence: Standard BS256 vs. Accumulated BS64x4")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Validation Accuracy trajectories
    ax2.plot(epochs_range, histories["Standard_BS256"]["val_acc"], label="Standard BS256", color="#1f77b4", marker="o", linewidth=2.2)
    ax2.plot(epochs_range, histories["Accumulated_BS64_x4"]["val_acc"], label="Accum BS64x4", color="#d62728", marker="s", linewidth=2.2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Validation Accuracy Trajectory")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(plot_dir, "task6_grad_accum_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[Task 6] Saved plot to {plot_path}")

    return summary_df


# ==============================================================================
# Task 7: Checkpoint Interruption and Resumption Study
# ==============================================================================

def run_checkpoint_resume_study(
    data_dir: str = "lab03/data",
    total_epochs: int = 6,
    interrupt_epoch: int = 3,
    batch_size: int = 128,
    device_str: str = "auto",
    seed: int = 42,
    output_dir: str = "lab06/results",
    checkpoint_dir: str = "lab06/checkpoints",
    plot_dir: str = "lab06/plots",
) -> pd.DataFrame:
    """
    Demonstrates interrupted and resumed training:
      1. Run A: Uninterrupted baseline (Epochs 1 to 6)
      2. Run B (Part 1): Train Epochs 1 to 3, save checkpoint
      3. Run B (Part 2): Resume from Epoch 3 checkpoint, train Epochs 4 to 6
      4. Compare training curves to verify smooth continuation.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    device = get_device(device_str)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=======================================================")
    print(f" Task 7: Checkpoint Interruption and Resumption Study")
    print(f" Total Epochs: {total_epochs} | Interrupt at Epoch: {interrupt_epoch}")
    print(f" Device: {device} | Batch Size: {batch_size}")
    print(f"=======================================================")

    # -------------------------------------------------------------------------
    # 1. Uninterrupted Baseline Run
    # -------------------------------------------------------------------------
    print("\n--- 1. Executing Uninterrupted Baseline Run (Epochs 1 -> 6) ---")
    set_seed(seed)
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
    )

    baseline_model = CNN().to(device)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)
    baseline_scheduler = optim.lr_scheduler.StepLR(baseline_optimizer, step_size=3, gamma=0.5)

    baseline_train_loss = []
    baseline_val_loss = []
    baseline_val_acc = []

    for epoch in range(1, total_epochs + 1):
        baseline_model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            baseline_optimizer.zero_grad()
            outputs = baseline_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            baseline_optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        if device.type == "mps":
            torch.mps.synchronize()

        ep_loss = running_loss / total_samples
        v_loss, v_acc = evaluate(baseline_model, val_loader, criterion, device)
        baseline_scheduler.step()

        baseline_train_loss.append(ep_loss)
        baseline_val_loss.append(v_loss)
        baseline_val_acc.append(v_acc)

        print(f"Baseline Epoch {epoch}/{total_epochs} | Train Loss: {ep_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:5.2f}%")

    base_test_loss, base_test_acc = evaluate(baseline_model, test_loader, criterion, device)

    # -------------------------------------------------------------------------
    # 2. Interrupted Run (Epochs 1 -> 3)
    # -------------------------------------------------------------------------
    print(f"\n--- 2. Executing Initial Training with Interruption (Epochs 1 -> {interrupt_epoch}) ---")
    set_seed(seed)
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
    )

    int_model = CNN().to(device)
    int_optimizer = optim.Adam(int_model.parameters(), lr=1e-3)
    int_scheduler = optim.lr_scheduler.StepLR(int_optimizer, step_size=3, gamma=0.5)

    resumed_train_loss = []
    resumed_val_loss = []
    resumed_val_acc = []

    for epoch in range(1, interrupt_epoch + 1):
        int_model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            int_optimizer.zero_grad()
            outputs = int_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            int_optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        if device.type == "mps":
            torch.mps.synchronize()

        ep_loss = running_loss / total_samples
        v_loss, v_acc = evaluate(int_model, val_loader, criterion, device)
        int_scheduler.step()

        resumed_train_loss.append(ep_loss)
        resumed_val_loss.append(v_loss)
        resumed_val_acc.append(v_acc)

        print(f"Interrupted Run Epoch {epoch}/{interrupt_epoch} | Train Loss: {ep_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:5.2f}%")

    # Save Checkpoint at Epoch 3
    checkpoint_file = os.path.join(checkpoint_dir, f"interrupted_epoch_{interrupt_epoch}.pt")
    save_checkpoint(
        checkpoint_file,
        epoch=interrupt_epoch,
        model=int_model,
        optimizer=int_optimizer,
        scheduler=int_scheduler,
        metrics={"train_loss": resumed_train_loss[-1], "val_loss": resumed_val_loss[-1], "val_acc": resumed_val_acc[-1]},
        config={"seed": seed, "batch_size": batch_size, "total_epochs": total_epochs},
    )
    print(f"[*] Checkpoint saved at epoch {interrupt_epoch} to {checkpoint_file}")

    # -------------------------------------------------------------------------
    # 3. Simulate Restart & Resumption (Epochs 4 -> 6)
    # -------------------------------------------------------------------------
    print(f"\n--- 3. Resuming from Checkpoint at Epoch {interrupt_epoch} (Epochs {interrupt_epoch + 1} -> {total_epochs}) ---")
    
    # Instantiate fresh model and optimizer
    resumed_model = CNN().to(device)
    resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = optim.lr_scheduler.StepLR(resumed_optimizer, step_size=3, gamma=0.5)

    # Restore from checkpoint
    chk = load_checkpoint(
        checkpoint_file,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        restore_rng=True,
        device=device,
    )
    start_epoch = chk["epoch"] + 1
    print(f"[*] Successfully restored model, optimizer, scheduler, and RNG state. Resuming at epoch {start_epoch}...")

    for epoch in range(start_epoch, total_epochs + 1):
        resumed_model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            resumed_optimizer.zero_grad()
            outputs = resumed_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            resumed_optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        if device.type == "mps":
            torch.mps.synchronize()

        ep_loss = running_loss / total_samples
        v_loss, v_acc = evaluate(resumed_model, val_loader, criterion, device)
        resumed_scheduler.step()

        resumed_train_loss.append(ep_loss)
        resumed_val_loss.append(v_loss)
        resumed_val_acc.append(v_acc)

        print(f"Resumed Run Epoch {epoch}/{total_epochs} | Train Loss: {ep_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:5.2f}%")

    resumed_test_loss, resumed_test_acc = evaluate(resumed_model, test_loader, criterion, device)

    # -------------------------------------------------------------------------
    # 4. Compare and Save Results
    # -------------------------------------------------------------------------
    records = []
    for ep in range(1, total_epochs + 1):
        idx = ep - 1
        records.append({
            "epoch": ep,
            "baseline_train_loss": baseline_train_loss[idx],
            "resumed_train_loss": resumed_train_loss[idx],
            "train_loss_diff": abs(baseline_train_loss[idx] - resumed_train_loss[idx]),
            "baseline_val_loss": baseline_val_loss[idx],
            "resumed_val_loss": resumed_val_loss[idx],
            "baseline_val_acc": baseline_val_acc[idx],
            "resumed_val_acc": resumed_val_acc[idx],
            "val_acc_diff": abs(baseline_val_acc[idx] - resumed_val_acc[idx]),
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "task7_checkpoint_resume.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Task 7] Saved comparison table to {csv_path}")

    # Plot Comparison Curves
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = list(range(1, total_epochs + 1))

    # Loss Curves
    ax1.plot(epochs_range, df["baseline_train_loss"], label="Uninterrupted (Train)", color="#1f77b4", marker="o", linewidth=2)
    ax1.plot(epochs_range, df["resumed_train_loss"], label="Interrupted & Resumed (Train)", color="#ff7f0e", marker="s", linestyle="--", linewidth=2)
    ax1.axvline(x=interrupt_epoch, color="#d62728", linestyle=":", label=f"Interruption Point (Epoch {interrupt_epoch})", linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss: Uninterrupted vs. Resumed")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy Curves
    ax2.plot(epochs_range, df["baseline_val_acc"], label="Uninterrupted (Val Acc)", color="#2ca02c", marker="o", linewidth=2)
    ax2.plot(epochs_range, df["resumed_val_acc"], label="Interrupted & Resumed (Val Acc)", color="#9467bd", marker="s", linestyle="--", linewidth=2)
    ax2.axvline(x=interrupt_epoch, color="#d62728", linestyle=":", label=f"Interruption Point (Epoch {interrupt_epoch})", linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Validation Accuracy: Uninterrupted vs. Resumed")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(plot_dir, "task7_resume_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[Task 7] Saved plot to {plot_path}")

    return df
