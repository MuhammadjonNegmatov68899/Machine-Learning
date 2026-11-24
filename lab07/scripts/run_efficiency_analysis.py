"""
Task 5: Four Pillars Efficiency Analysis Synthesis.
Compiles comprehensive experimental comparison across Data, Model, Compute, and Energy pillars.
Generates:
1. lab07/results/four_pillars_efficiency.csv
2. lab07/results/four_pillars_table.md
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
from lab07.src.efficiency import (
    generate_four_pillars_summary,
    format_markdown_four_pillars_table,
)


def main():
    results_dir = PROJECT_ROOT / "lab07" / "results"

    baseline_path = results_dir / "baseline_metrics.json"
    distill_path = results_dir / "distillation_comparison.json"
    global_path = results_dir / "global_pruning_results.csv"
    layer_path = results_dir / "layerwise_pruning_results.csv"
    struct_path = results_dir / "structured_pruning_results.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path}. Run train_baseline.py first!")
    if not distill_path.exists():
        raise FileNotFoundError(f"Missing {distill_path}. Run run_distillation_experiments.py first!")

    with open(baseline_path, "r") as f:
        baseline_metrics = json.load(f)

    with open(distill_path, "r") as f:
        distill_data = json.load(f)
        plain_student_metrics = distill_data["plain_student"]
        distill_student_metrics = distill_data["distilled_student"]

    # Pruned models list
    pruning_metrics = []

    if global_path.exists():
        gdf = pd.read_csv(global_path)
        for _, row in gdf.iterrows():
            if row["target_sparsity"] > 0.0:
                s_pct = int(round(row["target_sparsity"] * 100))
                pruning_metrics.append({
                    "name": f"Global Pruned {s_pct}%",
                    "category": "Global Magnitude",
                    "environment": "Apple M2 CPU",
                    "test_acc": row["post_finetune_test_acc"],
                    "total_params": int(row["total_params"]),
                    "nonzero_params": int(row["nonzero_params"]),
                    "sparsity_percent": row["actual_sparsity"] * 100.0,
                    "dense_size_mb": row["dense_size_mb"],
                    "sparse_coo_size_mb": row["sparse_coo_size_mb"],
                    "macs_m": baseline_metrics.get("macs_m", 0.0),  # Dense kernel computes all elements
                    "flops_m": baseline_metrics.get("flops_m", 0.0),
                    "finetune_time_sec": row["finetune_time_sec"],
                    "cpu_median_latency_ms": row["cpu_median_latency_ms"],
                    "cpu_throughput": row["cpu_throughput"],
                })

    if layer_path.exists():
        ldf = pd.read_csv(layer_path)
        for _, row in ldf.iterrows():
            if row["target_sparsity"] > 0.0:
                s_pct = int(round(row["target_sparsity"] * 100))
                env = "Apple M2 CPU" if s_pct <= 25 else "Kaggle Cloud vCPU"
                pruning_metrics.append({
                    "name": f"Layer-wise Pruned {s_pct}%",
                    "category": "Layer-wise Magnitude",
                    "environment": env,
                    "test_acc": row["post_finetune_test_acc"],
                    "total_params": int(row["total_params"]),
                    "nonzero_params": int(row["nonzero_params"]),
                    "sparsity_percent": row["actual_sparsity"] * 100.0,
                    "dense_size_mb": row["dense_size_mb"],
                    "sparse_coo_size_mb": row["sparse_coo_size_mb"],
                    "macs_m": baseline_metrics.get("macs_m", 0.0),
                    "flops_m": baseline_metrics.get("flops_m", 0.0),
                    "finetune_time_sec": row["finetune_time_sec"],
                    "cpu_median_latency_ms": row["cpu_median_latency_ms"],
                    "cpu_throughput": row["cpu_throughput"],
                })

    if struct_path.exists():
        sdf = pd.read_csv(struct_path)
        base_macs = baseline_metrics.get("macs_m", 1.0)
        base_flops = baseline_metrics.get("flops_m", 2.0)
        base_params = baseline_metrics.get("total_params", 1_011_866)

        for _, row in sdf.iterrows():
            if row["pruning_ratio"] == 0.0:
                pruning_metrics.append({
                    "name": "Baseline CNN (Kaggle Baseline)",
                    "category": "Baseline",
                    "environment": "Kaggle Cloud vCPU",
                    "test_acc": row["post_finetune_test_acc"],
                    "total_params": int(row["total_params"]),
                    "nonzero_params": int(row["nonzero_params"]),
                    "sparsity_percent": 0.0,
                    "dense_size_mb": row["dense_size_mb"],
                    "sparse_coo_size_mb": row["sparse_coo_size_mb"],
                    "macs_m": baseline_metrics.get("macs_m", 0.0),
                    "flops_m": baseline_metrics.get("flops_m", 0.0),
                    "finetune_time_sec": row["finetune_time_sec"],
                    "cpu_median_latency_ms": row["cpu_median_latency_ms"],
                    "cpu_throughput": row["cpu_throughput"],
                })
            else:
                r_pct = int(round(row["pruning_ratio"] * 100))
                param_factor = row["total_params"] / base_params
                pruning_metrics.append({
                    "name": f"Structured Pruned {r_pct}%",
                    "category": "Structured Filter",
                    "environment": "Kaggle Cloud vCPU",
                    "test_acc": row["post_finetune_test_acc"],
                    "total_params": int(row["total_params"]),
                    "nonzero_params": int(row["total_params"]),
                    "sparsity_percent": (1.0 - param_factor) * 100.0,
                    "dense_size_mb": row["dense_size_mb"],
                    "sparse_coo_size_mb": row["sparse_coo_size_mb"],
                    "macs_m": round(base_macs * param_factor, 2),
                    "flops_m": round(base_flops * param_factor, 2),
                    "finetune_time_sec": row["finetune_time_sec"],
                    "cpu_median_latency_ms": row["cpu_median_latency_ms"],
                    "cpu_throughput": row["cpu_throughput"],
                })

    csv_out = str(results_dir / "four_pillars_efficiency.csv")
    df = generate_four_pillars_summary(
        baseline_metrics=baseline_metrics,
        pruning_metrics=pruning_metrics,
        student_metrics=plain_student_metrics,
        distill_metrics=distill_student_metrics,
        output_csv=csv_out,
    )

    md_table = format_markdown_four_pillars_table(df)
    md_out = results_dir / "four_pillars_table.md"
    with open(md_out, "w") as f:
        f.write("# Chapter 9 Four Pillars Efficiency Summary\n\n")
        f.write(md_table)
        f.write("\n")

    print("\n" + md_table + "\n")
    print(f"Four pillars analysis saved to {csv_out} and {md_out}!")


if __name__ == "__main__":
    main()
