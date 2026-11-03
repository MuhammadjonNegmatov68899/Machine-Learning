#!/usr/bin/env python3
"""
Machine Learning Lab 05: Main Experiments Pipeline.

Executes all tasks end-to-end:
1. Task 4: Gradient Verification against PyTorch and Central Finite Differences
   for the expression ((a * b + c) ** 2).relu().sum() with tolerance < 1e-6.
2. Task 5: Model training on sklearn.datasets.make_moons across 5 random seeds,
   comparing Custom Engine vs PyTorch and reporting the median of 5 runs.
3. Task 6: PyTorch Framework Performance investigation:
   - Computation graph inspection.
   - Benchmarking eager vs compiled/traced execution.
   - Tensor contiguity inspection and performance benchmark.
4. Generates publication-ready benchmark plots and markdown summary.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import pandas as pd
import torch
import yaml

# Ensure lab05 root is in sys.path
LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from src.benchmarks import (
    generate_benchmark_plots,
    get_system_metadata,
    inspect_pytorch_graph,
    inspect_tensor_layout,
    run_execution_benchmarks,
    run_tensor_layout_benchmarks,
)
from src.neural_network import PyTorchMLP
from src.operations import (
    verify_expression_gradients,
    verify_vector_expression_gradients,
)
from src.training import run_5_fold_comparison


def load_config(config_path: Path) -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Lab 5 automatic differentiation and framework benchmarks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_DIR / "config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = LAB_DIR.parent
    results_dir = project_root / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" MACHINE LEARNING LAB 05: FRAMEWORKS & AUTOMATIC DIFFERENTIATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 0. Hardware and Environment Metadata
    # -------------------------------------------------------------------------
    meta = get_system_metadata()
    print("\n[Environment Metadata]")
    for k, v in meta.items():
        print(f"  - {k}: {v}")

    # -------------------------------------------------------------------------
    # Task 4: Gradient Verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 4: GRADIENT VERIFICATION")
    print("Expression: ((a * b + c) ** 2).relu().sum()")
    print("=" * 80)

    gv_cfg = config["gradient_verification"]
    scalar_df = verify_expression_gradients(
        a_val=gv_cfg["inputs"]["a"],
        b_val=gv_cfg["inputs"]["b"],
        c_val=gv_cfg["inputs"]["c"],
        eps=float(gv_cfg["finite_diff_epsilon"]),
        tol=float(gv_cfg["tolerance"]),
    )

    vector_df = verify_vector_expression_gradients(
        a_vec=gv_cfg["vector_test"]["a"],
        b_vec=gv_cfg["vector_test"]["b"],
        c_vec=gv_cfg["vector_test"]["c"],
        eps=float(gv_cfg["finite_diff_epsilon"]),
        tol=float(gv_cfg["tolerance"]),
    )

    combined_gradient_df = pd.concat([scalar_df, vector_df], ignore_index=True)
    grad_csv_path = project_root / config["paths"]["gradient_comparison_csv"]
    combined_gradient_df.to_csv(grad_csv_path, index=False)
    print(f"\nGradient Verification Results saved to: {grad_csv_path}")
    print(scalar_df.to_string(index=False))

    max_scalar_err = scalar_df["max_abs_error"].max()
    max_vector_err = vector_df["max_abs_error"].max()
    overall_max_err = max(max_scalar_err, max_vector_err)

    print(f"\nMaximum Absolute Error (Scalar): {max_scalar_err:.3e}")
    print(f"Maximum Absolute Error (Vector): {max_vector_err:.3e}")
    print(f"Tolerance Threshold: {gv_cfg['tolerance']:.1e}")

    assert overall_max_err < float(gv_cfg["tolerance"]), (
        f"Gradient error {overall_max_err:.3e} exceeds tolerance {gv_cfg['tolerance']}"
    )
    print(">>> TASK 4 STATUS: ALL GRADIENT CHECKS PASSED (< 1e-6) <<<")

    # -------------------------------------------------------------------------
    # Task 5: Train Tiny Model on make_moons
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 5: TINY MODEL TRAINING ON MAKE_MOONS (2 -> 8 -> 1)")
    print("Architecture: Sigmoid Activation | Loss: Binary Cross-Entropy")
    print("=" * 80)

    tr_cfg = config["training"]
    training_df, training_plot_data = run_5_fold_comparison(
        seeds=tr_cfg["seeds"],
        epochs=tr_cfg["hyperparameters"]["epochs"],
        lr=tr_cfg["hyperparameters"]["learning_rate"],
        n_samples=tr_cfg["dataset"]["n_samples"],
        noise=tr_cfg["dataset"]["noise"],
        test_size=tr_cfg["dataset"]["test_size"],
    )

    train_csv_path = project_root / config["paths"]["training_comparison_csv"]
    training_df.to_csv(train_csv_path, index=False)
    print(f"\nTraining Comparison Results saved to: {train_csv_path}")
    print(training_df.to_string(index=False))

    custom_median = training_df[
        (training_df["run_id"] == "MEDIAN") & (training_df["engine"] == "custom")
    ].iloc[0]
    torch_median = training_df[
        (training_df["run_id"] == "MEDIAN") & (training_df["engine"] == "pytorch")
    ].iloc[0]

    print("\n[Median of 5 Runs Summary]")
    print(
        f"  - Custom Engine: Train Loss = {custom_median['train_loss']:.4f}, "
        f"Test Loss = {custom_median['test_loss']:.4f}, "
        f"Train Acc = {custom_median['train_acc']:.2%}, "
        f"Test Acc = {custom_median['test_acc']:.2%}, "
        f"Time = {custom_median['train_time_sec']:.3f}s"
    )
    print(
        f"  - PyTorch Engine: Train Loss = {torch_median['train_loss']:.4f}, "
        f"Test Loss = {torch_median['test_loss']:.4f}, "
        f"Train Acc = {torch_median['train_acc']:.2%}, "
        f"Test Acc = {torch_median['test_acc']:.2%}, "
        f"Time = {torch_median['train_time_sec']:.3f}s"
    )
    print(">>> TASK 5 STATUS: COMPLETED SUCCESSFULLY <<<")

    # -------------------------------------------------------------------------
    # Task 6: PyTorch Framework Performance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 6: PYTORCH FRAMEWORK PERFORMANCE & GRAPH INSPECTION")
    print("=" * 80)

    # 1. Computation Graph Inspection
    print("\n[1. PyTorch Computation Graph Inspection]")
    sample_model = PyTorchMLP(nin=2, hidden_dim=8, nout=1, hidden_activation="sigmoid", seed=42)
    sample_in = torch.randn(4, 2)
    graph_info = inspect_pytorch_graph(sample_model, sample_in)
    print("Backward Graph (grad_fn chain):")
    print(graph_info["backward_graph_text"])
    print("\nTorchScript Intermediate Representation (IR):")
    print(graph_info["torchscript_code"])

    # 2. Execution Benchmarks
    print("\n[2. Eager vs Compiled / Traced Benchmarks]")
    bench_cfg = config["benchmarks"]["framework"]
    exec_df = run_execution_benchmarks(
        batch_size=bench_cfg["batch_size"],
        input_dim=bench_cfg["input_dim"],
        hidden_dim=bench_cfg["hidden_dim"],
        output_dim=bench_cfg["output_dim"],
        warmup_iters=bench_cfg["warmup_iterations"],
        timed_iters=bench_cfg["timed_iterations"],
    )
    exec_csv_path = project_root / config["paths"]["execution_benchmarks_csv"]
    exec_df.to_csv(exec_csv_path, index=False)
    print(f"Execution Benchmarks saved to: {exec_csv_path}")
    print(exec_df[["mode", "status", "median_latency_us", "throughput_inferences_per_sec", "speedup_vs_eager"]].to_string(index=False))

    # 3. Tensor Contiguity & Memory Layout Benchmarks
    print("\n[3. Tensor Memory Layout & Contiguity Benchmarks]")
    layout_cfg = config["benchmarks"]["tensor_layout"]
    layout_df, layout_meta = run_tensor_layout_benchmarks(
        rows=layout_cfg["matrix_rows"],
        cols=layout_cfg["matrix_cols"],
        warmup_iters=layout_cfg["warmup_iterations"],
        timed_iters=layout_cfg["timed_iterations"],
        dtype=torch.float32,
    )
    layout_csv_path = project_root / config["paths"]["tensor_layout_benchmarks_csv"]
    layout_df.to_csv(layout_csv_path, index=False)
    print(f"Tensor Layout Benchmarks saved to: {layout_csv_path}")
    print(layout_df[["operation", "contiguous_stride", "noncontiguous_stride", "contiguous_median_ms", "noncontiguous_median_ms", "slowdown_factor"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # Visualization Generation
    # -------------------------------------------------------------------------
    plot_path = project_root / config["paths"]["benchmark_plots_png"]
    generate_benchmark_plots(training_plot_data, exec_df, layout_df, str(plot_path))
    print(f"\nGenerated Multi-Panel Benchmark Plot: {plot_path}")

    # -------------------------------------------------------------------------
    # Markdown Summary Generation
    # -------------------------------------------------------------------------
    summary_path = project_root / config["paths"]["summary_md"]
    generate_summary_markdown(
        summary_path=summary_path,
        meta=meta,
        gradient_df=scalar_df,
        training_df=training_df,
        exec_df=exec_df,
        layout_df=layout_df,
        graph_info=graph_info,
        layout_meta=layout_meta,
    )
    print(f"Generated Comprehensive Summary: {summary_path}")

    print("\n" + "=" * 80)
    print(" ALL LAB 05 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return 0


def generate_summary_markdown(
    summary_path: Path,
    meta: dict,
    gradient_df: pd.DataFrame,
    training_df: pd.DataFrame,
    exec_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    graph_info: dict,
    layout_meta: dict | None = None,
) -> None:
    """Write comprehensive experimental summary in Markdown format."""
    custom_median = training_df[
        (training_df["run_id"] == "MEDIAN") & (training_df["engine"] == "custom")
    ].iloc[0]
    torch_median = training_df[
        (training_df["run_id"] == "MEDIAN") & (training_df["engine"] == "pytorch")
    ].iloc[0]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Machine Learning Lab 05: Experimental Summary\n\n")

        f.write("## 1. System and Environment Metadata\n\n")
        f.write("| Attribute | Value |\n|---|---|\n")
        for k, v in meta.items():
            f.write(f"| `{k}` | `{v}` |\n")

        f.write("\n## 2. Task 4: Gradient Verification Results\n\n")
        f.write("**Target Expression:** `((a * b + c) ** 2).relu().sum()`\n\n")
        f.write(gradient_df.to_markdown(index=False))
        f.write(f"\n\n- **Verification Verdict:** Strict Pass (< 1e-6 error across all variables).\n")

        f.write("\n## 3. Task 5: make_moons Training Parity (2 -> 8 -> 1 MLP)\n\n")
        f.write("### Multi-Seed Results Table\n\n")
        f.write(training_df.to_markdown(index=False))

        f.write("\n\n### Median of 5 Runs Comparison\n\n")
        f.write("| Metric | Custom AD Engine | PyTorch Engine | Disparity |\n|---|---|---|---|\n")
        f.write(f"| **Final Train Loss** | {custom_median['train_loss']:.4f} | {torch_median['train_loss']:.4f} | {abs(custom_median['train_loss'] - torch_median['train_loss']):.4f} |\n")
        f.write(f"| **Final Test Loss** | {custom_median['test_loss']:.4f} | {torch_median['test_loss']:.4f} | {abs(custom_median['test_loss'] - torch_median['test_loss']):.4f} |\n")
        f.write(f"| **Train Accuracy** | {custom_median['train_acc']:.2%} | {torch_median['train_acc']:.2%} | {abs(custom_median['train_acc'] - torch_median['train_acc']):.2%} |\n")
        f.write(f"| **Test Accuracy** | {custom_median['test_acc']:.2%} | {torch_median['test_acc']:.2%} | {abs(custom_median['test_acc'] - torch_median['test_acc']):.2%} |\n")
        f.write(f"| **Training Duration** | {custom_median['train_time_sec']:.3f} s | {torch_median['train_time_sec']:.3f} s | {custom_median['train_time_sec'] / torch_median['train_time_sec']:.1f}x slower |\n")

        f.write("\n## 4. Task 6: Framework Execution Benchmarks\n\n")
        f.write(exec_df.to_markdown(index=False))

        f.write("\n\n## 5. Task 6: Tensor Layout & Contiguity Benchmarks\n\n")
        if layout_meta:
            c_meta = layout_meta.get("contiguous", {})
            nc_meta = layout_meta.get("noncontiguous", {})
            f.write("### Tensor Layout Properties (Inspection)\n\n")
            f.write("| Property | Contiguous Tensor (`x_contig`) | Transposed View (`x_noncontig`) |\n")
            f.write("|---|---|---|\n")
            f.write(f"| **Name** | `{c_meta.get('name', 'N/A')}` | `{nc_meta.get('name', 'N/A')}` |\n")
            f.write(f"| **Shape** | `{c_meta.get('shape', 'N/A')}` | `{nc_meta.get('shape', 'N/A')}` |\n")
            f.write(f"| **Dtype** | `{c_meta.get('dtype', 'N/A')}` | `{nc_meta.get('dtype', 'N/A')}` |\n")
            f.write(f"| **Stride** | `{c_meta.get('stride', 'N/A')}` | `{nc_meta.get('stride', 'N/A')}` |\n")
            f.write(f"| **Contiguous (`is_contiguous`)** | `{c_meta.get('is_contiguous', 'N/A')}` | `{nc_meta.get('is_contiguous', 'N/A')}` |\n")
            f.write(f"| **Element Size (bytes)** | `{c_meta.get('element_size_bytes', 'N/A')}` | `{nc_meta.get('element_size_bytes', 'N/A')}` |\n")
            f.write(f"| **Total Elements** | `{c_meta.get('total_elements', 'N/A')}` | `{nc_meta.get('total_elements', 'N/A')}` |\n")
            f.write(f"| **Total Memory (KB)** | `{c_meta.get('total_memory_kb', 'N/A'):.1f} KB` | `{nc_meta.get('total_memory_kb', 'N/A'):.1f} KB` |\n\n")
            f.write("### Benchmark Results\n\n")

        f.write(layout_df.to_markdown(index=False))

        f.write("\n\n## 6. Computation Graph Inspection\n\n")
        f.write("```text\n")
        f.write(graph_info["backward_graph_text"])
        f.write("\n```\n")


if __name__ == "__main__":
    sys.exit(main())
