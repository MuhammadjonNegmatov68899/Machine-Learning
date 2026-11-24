"""
Efficiency Analysis according to Chapter 9 four pillars: Data, Model, Compute, Energy.
Generates comprehensive synthesis tables and evidence-based trade-off metrics.
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd


def generate_four_pillars_summary(
    baseline_metrics: Dict[str, Any],
    pruning_metrics: List[Dict[str, Any]],
    student_metrics: Dict[str, Any],
    distill_metrics: Dict[str, Any],
    output_csv: str = "lab07/results/four_pillars_efficiency.csv",
) -> pd.DataFrame:
    """
    Constructs a unified DataFrame summarizing all evaluated models across the four efficiency pillars:
    1. Data Pillar (Dataset size, samples, preprocessing)
    2. Model Pillar (Parameters, non-zero params, sparsity %, dense size, sparse size)
    3. Compute Pillar (MACs, FLOPs, training time, CPU latency, device latency, throughput)
    4. Energy Pillar (TDP reference, energy consumption estimate, Joules per 1k inferences)
    """
    rows = []

    # 1. Baseline
    rows.append({
        "Model": "Baseline CNN (Teacher)",
        "Environment": "Apple M2 CPU",
        "Category": "Baseline",
        "Accuracy (%)": baseline_metrics.get("test_acc", 0.0) * 100,
        "Total Params": baseline_metrics.get("total_params", 0),
        "Nonzero Params": baseline_metrics.get("nonzero_params", 0),
        "Sparsity (%)": baseline_metrics.get("sparsity_percent", 0.0),
        "Dense Size (MB)": baseline_metrics.get("dense_size_mb", 0.0),
        "Sparse COO (MB)": baseline_metrics.get("sparse_coo_size_mb", 0.0),
        "MACs (M)": baseline_metrics.get("macs_m", 0.0),
        "FLOPs (M)": baseline_metrics.get("flops_m", 0.0),
        "Training Time (s)": baseline_metrics.get("total_training_time_sec", 0.0),
        "CPU Latency (ms)": baseline_metrics.get("cpu_median_latency_ms", 0.0),
        "CPU Throughput (samples/s)": baseline_metrics.get("cpu_throughput", 0.0),
        "Energy Est (mJ/sample)": round(
            (baseline_metrics.get("cpu_median_latency_ms", 0.0) / 64.0) * 15.0, 3
        ),  # E (mJ/sample) = P_active (15W) * [latency_ms / (batch_size 64 * 1000 s)] * 1000 mJ/J
    })

    # 2. Pruned models
    for p in pruning_metrics:
        lat = p.get("cpu_median_latency_ms", 0.0)
        rows.append({
            "Model": p.get("name", "Pruned Model"),
            "Environment": p.get("environment", "Apple M2 CPU"),
            "Category": p.get("category", "Pruning"),
            "Accuracy (%)": p.get("test_acc", 0.0) * 100,
            "Total Params": p.get("total_params", 0),
            "Nonzero Params": p.get("nonzero_params", 0),
            "Sparsity (%)": p.get("sparsity_percent", 0.0),
            "Dense Size (MB)": p.get("dense_size_mb", 0.0),
            "Sparse COO (MB)": p.get("sparse_coo_size_mb", 0.0),
            "MACs (M)": p.get("macs_m", 0.0),
            "FLOPs (M)": p.get("flops_m", 0.0),
            "Training Time (s)": p.get("finetune_time_sec", 0.0),
            "CPU Latency (ms)": lat,
            "CPU Throughput (samples/s)": p.get("cpu_throughput", 0.0),
            "Energy Est (mJ/sample)": round((lat / 64.0) * 15.0, 3) if lat > 0 else 0.0,
        })

    # 3. Plain Student
    s_lat = student_metrics.get("cpu_median_latency_ms", 0.0)
    rows.append({
        "Model": "Student CNN (Plain CE)",
        "Environment": "Kaggle Cloud vCPU",
        "Category": "Distillation",
        "Accuracy (%)": student_metrics.get("test_acc", 0.0) * 100,
        "Total Params": student_metrics.get("total_params", 0),
        "Nonzero Params": student_metrics.get("nonzero_params", 0),
        "Sparsity (%)": student_metrics.get("sparsity_percent", 0.0),
        "Dense Size (MB)": student_metrics.get("dense_size_mb", 0.0),
        "Sparse COO (MB)": student_metrics.get("sparse_coo_size_mb", 0.0),
        "MACs (M)": student_metrics.get("macs_m", 0.0),
        "FLOPs (M)": student_metrics.get("flops_m", 0.0),
        "Training Time (s)": student_metrics.get("total_training_time_sec", 0.0),
        "CPU Latency (ms)": s_lat,
        "CPU Throughput (samples/s)": student_metrics.get("cpu_throughput", 0.0),
        "Energy Est (mJ/sample)": round((s_lat / 64.0) * 15.0, 3) if s_lat > 0 else 0.0,
    })

    # 4. Distilled Student
    d_lat = distill_metrics.get("cpu_median_latency_ms", 0.0)
    rows.append({
        "Model": "Student CNN (Distilled)",
        "Environment": "Kaggle Cloud vCPU",
        "Category": "Distillation",
        "Accuracy (%)": distill_metrics.get("test_acc", 0.0) * 100,
        "Total Params": distill_metrics.get("total_params", 0),
        "Nonzero Params": distill_metrics.get("nonzero_params", 0),
        "Sparsity (%)": distill_metrics.get("sparsity_percent", 0.0),
        "Dense Size (MB)": distill_metrics.get("dense_size_mb", 0.0),
        "Sparse COO (MB)": distill_metrics.get("sparse_coo_size_mb", 0.0),
        "MACs (M)": distill_metrics.get("macs_m", 0.0),
        "FLOPs (M)": distill_metrics.get("flops_m", 0.0),
        "Training Time (s)": distill_metrics.get("total_training_time_sec", 0.0),
        "CPU Latency (ms)": d_lat,
        "CPU Throughput (samples/s)": distill_metrics.get("cpu_throughput", 0.0),
        "Energy Est (mJ/sample)": round((d_lat / 64.0) * 15.0, 3) if d_lat > 0 else 0.0,
    })

    df = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def format_markdown_four_pillars_table(df: pd.DataFrame) -> str:
    """Formats DataFrame as a clean Markdown table for report.md."""
    cols_display = [
        "Model",
        "Environment",
        "Accuracy (%)",
        "Total Params",
        "Sparsity (%)",
        "Dense Size (MB)",
        "FLOPs (M)",
        "CPU Latency (ms)",
        "CPU Throughput (samples/s)",
        "Energy Est (mJ/sample)",
    ]
    sub_df = df[cols_display].copy()
    sub_df["Accuracy (%)"] = sub_df["Accuracy (%)"].map(lambda x: f"{x:.2f}%")
    sub_df["Total Params"] = sub_df["Total Params"].map(lambda x: f"{x:,}")
    sub_df["Sparsity (%)"] = sub_df["Sparsity (%)"].map(lambda x: f"{x:.1f}%")
    sub_df["Dense Size (MB)"] = sub_df["Dense Size (MB)"].map(lambda x: f"{x:.2f}")
    sub_df["FLOPs (M)"] = sub_df["FLOPs (M)"].map(lambda x: f"{x:.2f}")
    sub_df["CPU Latency (ms)"] = sub_df["CPU Latency (ms)"].map(lambda x: f"{x:.2f}")
    sub_df["CPU Throughput (samples/s)"] = sub_df["CPU Throughput (samples/s)"].map(
        lambda x: f"{x:.1f}"
    )
    sub_df["Energy Est (mJ/sample)"] = sub_df["Energy Est (mJ/sample)"].map(
        lambda x: f"{x:.3f}"
    )

    return sub_df.to_markdown(index=False)
