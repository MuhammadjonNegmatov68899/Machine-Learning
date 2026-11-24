"""
Publication-quality plotting and visualization routines for Lab 07.
Generates:
1. Global Pruning Accuracy vs Sparsity Curve
2. Layer-wise Pruning Accuracy vs Sparsity Curve
3. Structured Filter Pruning Accuracy vs Pruning Ratio Curve
4. Multi-method Pruning Comparison (Global vs Layerwise vs Structured)
5. Dense vs Sparse Storage and Hardware Latency Analysis
6. Knowledge Distillation vs Plain Student Training Comparison
"""

import os
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def setup_plot_style():
    """Sets a clean, publication-ready plot aesthetic."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })


def plot_global_pruning(
    results_df: pd.DataFrame, output_path: str = "lab07/plots/global_pruning_curve.png"
):
    """Plots accuracy vs sparsity for Global Unstructured Magnitude Pruning."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        results_df["target_sparsity"] * 100,
        results_df["pre_finetune_test_acc"] * 100,
        marker="o",
        linestyle="--",
        color="#e74c3c",
        label="Pre-Finetune (Zero-Shot Pruned)",
        linewidth=2,
    )
    ax.plot(
        results_df["target_sparsity"] * 100,
        results_df["post_finetune_test_acc"] * 100,
        marker="s",
        linestyle="-",
        color="#2980b9",
        label="Post-Finetune (3 Epochs Fixed Mask)",
        linewidth=2.5,
    )

    ax.set_xlabel("Global Weight Sparsity (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Global Unstructured Magnitude Pruning: Accuracy vs Sparsity")
    ax.set_ylim(bottom=max(0, results_df["post_finetune_test_acc"].min() * 100 - 15), top=100)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Highlight baseline
    baseline_acc = results_df.iloc[0]["post_finetune_test_acc"] * 100
    ax.axhline(baseline_acc, color="#7f8c8d", linestyle=":", alpha=0.7, label="Baseline")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_layerwise_pruning(
    results_df: pd.DataFrame, output_path: str = "lab07/plots/layerwise_pruning_curve.png"
):
    """Plots accuracy vs sparsity for Layer-wise Unstructured Magnitude Pruning."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        results_df["target_sparsity"] * 100,
        results_df["pre_finetune_test_acc"] * 100,
        marker="o",
        linestyle="--",
        color="#e67e22",
        label="Pre-Finetune (Zero-Shot Pruned)",
        linewidth=2,
    )
    ax.plot(
        results_df["target_sparsity"] * 100,
        results_df["post_finetune_test_acc"] * 100,
        marker="s",
        linestyle="-",
        color="#27ae60",
        label="Post-Finetune (3 Epochs Fixed Mask)",
        linewidth=2.5,
    )

    ax.set_xlabel("Layer-wise Weight Sparsity (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Layer-wise Unstructured Magnitude Pruning: Accuracy vs Sparsity")
    ax.set_ylim(bottom=max(0, results_df["post_finetune_test_acc"].min() * 100 - 15), top=100)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_structured_pruning(
    results_df: pd.DataFrame, output_path: str = "lab07/plots/structured_pruning_curve.png"
):
    """Plots accuracy vs filter pruning ratio for Structured Filter Pruning."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        results_df["pruning_ratio"] * 100,
        results_df["pre_finetune_test_acc"] * 100,
        marker="o",
        linestyle="--",
        color="#8e44ad",
        label="Pre-Finetune (Zero-Shot Contraction)",
        linewidth=2,
    )
    ax.plot(
        results_df["pruning_ratio"] * 100,
        results_df["post_finetune_test_acc"] * 100,
        marker="^",
        linestyle="-",
        color="#d35400",
        label="Post-Finetune (3 Epochs)",
        linewidth=2.5,
    )

    ax.set_xlabel("Filter Pruning Ratio (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Structured Filter Pruning (L2 Norm): Accuracy vs Filter Ratio")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_combined_pruning_comparison(
    global_df: pd.DataFrame,
    layerwise_df: pd.DataFrame,
    structured_df: pd.DataFrame,
    output_path: str = "lab07/plots/pruning_methods_comparison.png",
):
    """Comparative plot showing post-finetune accuracy across all three pruning strategies."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        global_df["target_sparsity"] * 100,
        global_df["post_finetune_test_acc"] * 100,
        marker="s",
        linewidth=2.5,
        color="#2980b9",
        label="Global Magnitude Pruning (Unstructured)",
    )
    ax.plot(
        layerwise_df["target_sparsity"] * 100,
        layerwise_df["post_finetune_test_acc"] * 100,
        marker="o",
        linewidth=2.5,
        color="#27ae60",
        label="Layer-wise Magnitude Pruning (Unstructured)",
    )

    # For structured pruning, calculate actual parameter sparsity for fair comparison on x-axis
    baseline_params = structured_df.iloc[0]["total_params"]
    struct_param_sparsity = (
        (1.0 - structured_df["total_params"] / baseline_params) * 100.0
    )
    ax.plot(
        struct_param_sparsity,
        structured_df["post_finetune_test_acc"] * 100,
        marker="^",
        linewidth=2.5,
        color="#d35400",
        label="Structured Filter Pruning (Actual Param Reduction)",
    )

    ax.set_xlabel("Effective Parameter Sparsity / Reduction (%)")
    ax.set_ylabel("Post-Finetune Test Accuracy (%)")
    ax.set_title("Pruning Strategies Comparison: Accuracy Retention vs Parameter Sparsity")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_storage_and_latency_analysis(
    storage_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    output_path: str = "lab07/plots/storage_and_latency_tradeoffs.png",
):
    """Plots storage footprint (dense vs sparse) and inference latency on CPU."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Storage Comparison
    ax1.plot(
        storage_df["sparsity"] * 100,
        storage_df["dense_size_kb"],
        marker="o",
        color="#2c3e50",
        label="Dense Format (.pt)",
        linewidth=2,
    )
    ax1.plot(
        storage_df["sparsity"] * 100,
        storage_df["compressed_size_kb"],
        marker="s",
        color="#16a085",
        label="Compressed Dense (.pt.gz)",
        linewidth=2,
    )
    ax1.plot(
        storage_df["sparsity"] * 100,
        storage_df["sparse_coo_size_kb"],
        marker="^",
        color="#c0392b",
        label="Sparse Coordinate (COO)",
        linewidth=2,
    )
    ax1.set_xlabel("Sparsity (%)")
    ax1.set_ylabel("Storage on Disk (KB)")
    ax1.set_title("Storage Footprint: Dense vs Compressed vs Sparse COO")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Latency Comparison (Unstructured vs Structured)
    ax2.plot(
        latency_df["sparsity"] * 100,
        latency_df["unstructured_cpu_ms"],
        marker="s",
        color="#2980b9",
        label="Unstructured Sparsity (Dense CPU Kernel)",
        linewidth=2.5,
    )
    ax2.plot(
        latency_df["structured_param_reduction"] * 100,
        latency_df["structured_cpu_ms"],
        marker="^",
        color="#e67e22",
        label="Structured Filter Pruning (Smaller Model)",
        linewidth=2.5,
    )
    ax2.set_xlabel("Sparsity / Parameter Reduction (%)")
    ax2.set_ylabel("Inference Latency (ms/batch)")
    ax2.set_title("Hardware CPU Latency: Unstructured vs Structured")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_distillation_comparison(
    plain_history: List[Dict[str, Any]],
    distill_history: List[Dict[str, Any]],
    output_path: str = "lab07/plots/knowledge_distillation_comparison.png",
):
    """Plots training and validation accuracy curves for student with and without distillation."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs = [h["epoch"] for h in plain_history]

    # Accuracy curves
    ax1.plot(
        epochs,
        [h["val_acc"] * 100 for h in plain_history],
        marker="o",
        color="#e74c3c",
        label="Plain Student (CE Only)",
        linewidth=2,
    )
    ax1.plot(
        epochs,
        [h["val_acc"] * 100 for h in distill_history],
        marker="s",
        color="#2980b9",
        label="Distilled Student (Teacher = Baseline)",
        linewidth=2.5,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Knowledge Distillation: Validation Accuracy Trajectory")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss curves
    ax2.plot(
        epochs,
        [h["train_loss"] for h in plain_history],
        marker="o",
        color="#e74c3c",
        label="Plain Student Train Loss",
        linewidth=2,
    )
    ax2.plot(
        epochs,
        [h["train_loss"] for h in distill_history],
        marker="s",
        color="#2980b9",
        label="Distilled Student Total Loss",
        linewidth=2.5,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss Trajectory")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
