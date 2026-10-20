"""
Plotting module to visualize accuracy vs. parameters, accuracy vs. latency,
and training dynamics for Lab 3.
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_parameters(
    summary_data: List[Dict[str, Any]],
    output_path: str = "results/accuracy_vs_params.png",
) -> None:
    """
    Plots Test Accuracy (%) against Total Parameters.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)

    # Style mapping
    colors = {"MLP": "#1f77b4", "CNN": "#2ca02c", "Attention": "#d62728"}
    markers = {"specified": "o", "calibrated": "s"}

    for item in summary_data:
        arch = item["model"]
        mode = item.get("mode", "specified")
        params_k = item["params"] / 1000.0
        acc = item["test_acc"]
        label = f"{arch} ({mode})"
        color = colors.get(arch, "#333333")
        marker = markers.get(mode, "o")

        plt.scatter(
            params_k,
            acc,
            color=color,
            marker=marker,
            s=130,
            edgecolors="black",
            linewidth=1.2,
            zorder=4,
            label=label,
        )
        offset_y = 0.3 if arch != "CNN" else -0.5
        plt.annotate(
            f"{arch}\n({params_k:.1f}k, {acc:.2f}%)",
            (params_k, acc),
            textcoords="offset points",
            xytext=(10, offset_y * 10),
            fontsize=9,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    plt.axvspan(100, 150, color="gray", alpha=0.15, label="Target Range (100k–150k params)")

    plt.title("Test Accuracy vs. Parameter Count", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Total Parameters (thousands)", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_latency(
    summary_data: List[Dict[str, Any]],
    output_path: str = "results/accuracy_vs_latency.png",
) -> None:
    """
    Plots Test Accuracy (%) against Inference Latency (Batch 1 and Batch 128).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    colors = {"MLP": "#1f77b4", "CNN": "#2ca02c", "Attention": "#d62728"}
    markers = {"specified": "o", "calibrated": "s"}

    for item in summary_data:
        arch = item["model"]
        mode = item.get("mode", "specified")
        acc = item["test_acc"]
        lat_b1 = item["latency_b1_ms"]
        lat_b128 = item["latency_b128_ms"]
        color = colors.get(arch, "#333333")
        marker = markers.get(mode, "o")
        label = f"{arch} ({mode})"

        # Subplot 1: Batch size 1
        ax1.scatter(lat_b1, acc, color=color, marker=marker, s=120, edgecolors="black", linewidth=1.2, zorder=4, label=label)
        ax1.annotate(
            f"{arch}\n({lat_b1:.3f} ms, {acc:.1f}%)",
            (lat_b1, acc),
            textcoords="offset points",
            xytext=(8, -5),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
        )

        # Subplot 2: Batch size 128
        ax2.scatter(lat_b128, acc, color=color, marker=marker, s=120, edgecolors="black", linewidth=1.2, zorder=4, label=label)
        ax2.annotate(
            f"{arch}\n({lat_b128:.3f} ms, {acc:.1f}%)",
            (lat_b128, acc),
            textcoords="offset points",
            xytext=(8, -5),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
        )

    ax1.set_title("Inference Latency: Batch Size = 1 (100 runs)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Median Latency (ms)", fontsize=10)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, fontsize=8.5, loc="lower right")

    ax2.set_title("Inference Latency: Batch Size = 128 (20 runs)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Median Latency (ms)", fontsize=10)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, fontsize=8.5, loc="lower right")

    plt.suptitle("Accuracy vs. Inference Latency Trade-Off", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    output_path: str = "results/training_curves.png",
) -> None:
    """
    Plots training loss and validation accuracy curves over the 5 epochs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    color_cycle = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]

    for i, (name, hist) in enumerate(histories.items()):
        color = color_cycle[i % len(color_cycle)]
        epochs = hist["epoch"]

        # Loss curve (solid: train loss, dashed: val loss)
        ax1.plot(epochs, hist["train_loss"], label=f"{name} (Train)", color=color, linewidth=2)
        ax1.plot(epochs, hist["val_loss"], label=f"{name} (Val)", color=color, linestyle="--", linewidth=1.8)

        # Validation accuracy curve
        ax2.plot(epochs, hist["val_acc"], label=f"{name}", color=color, marker="o", linewidth=2)

    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_xticks(range(1, 6))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, fontsize=8)

    ax2.set_title("Validation Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_xticks(range(1, 6))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, fontsize=8.5, loc="lower right")

    plt.suptitle("Training Dynamics Across Architectures (5 Epochs, Adam, lr=1e-3)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
