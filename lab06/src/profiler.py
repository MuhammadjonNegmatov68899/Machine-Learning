"""
PyTorch Profiling Module for Lab 6 (Task 2).
Uses torch.profiler to measure exact time and percentage spent on:
- Data loading
- Forward pass
- Backward pass
- Optimizer step
"""

from typing import Dict, Any, Tuple
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from .models import CNN
from .dataset import get_fashion_mnist_dataloaders
from .utils import set_seed, get_device, setup_plot_style


def profile_training_epoch(
    data_dir: str = "lab03/data",
    batch_size: int = 128,
    num_workers: int = 0,
    device_str: str = "auto",
    seed: int = 42,
    export_trace: bool = True,
    output_dir: str = "lab06/results",
    plot_dir: str = "lab06/plots",
) -> Dict[str, Any]:
    """
    Profiles one complete training epoch using torch.profiler and high-precision timers,
    breaking down execution time into:
      1. Data Loading
      2. Forward Pass
      3. Backward Pass
      4. Optimizer Step

    Returns:
        Dictionary containing timing metrics, percentages, and export paths.
    """
    set_seed(seed)
    device = get_device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n[Task 2 Profiler] Setting up profiling on device: {device} (batch_size={batch_size}, num_workers={num_workers})...", flush=True)
    train_loader, _, _ = get_fashion_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # Accumulators for exact wall-clock time across the full epoch
    time_data_loading = 0.0
    time_forward = 0.0
    time_backward = 0.0
    time_optimizer = 0.0

    num_batches = len(train_loader)
    data_iter = iter(train_loader)

    print(f"[Task 2 Profiler] Profiling {num_batches} batches with torch.profiler (optimized for M2 hardware)...", flush=True)

    # Use standard profiler schedule: warmup then capture active trace
    schedule = torch.profiler.schedule(wait=2, warmup=3, active=25, repeat=1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        optimizer.zero_grad()

        for step in range(num_batches):
            # 1. Data loading phase
            t0 = time.perf_counter()
            with torch.profiler.record_function("data_loading"):
                inputs, targets = next(data_iter)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            time_data_loading += (t1 - t0)

            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # 2. Forward pass phase
            t2 = time.perf_counter()
            with torch.profiler.record_function("forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if device.type == "mps":
                torch.mps.synchronize()
            t3 = time.perf_counter()
            time_forward += (t3 - t2)

            # 3. Backward pass phase
            t4 = time.perf_counter()
            with torch.profiler.record_function("backward_pass"):
                loss.backward()
            if device.type == "mps":
                torch.mps.synchronize()
            t5 = time.perf_counter()
            time_backward += (t5 - t4)

            # 4. Optimizer step phase
            t6 = time.perf_counter()
            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
            if device.type == "mps":
                torch.mps.synchronize()
            t7 = time.perf_counter()
            time_optimizer += (t7 - t6)

            prof.step()

    total_time = time_data_loading + time_forward + time_backward + time_optimizer

    pct_data = (time_data_loading / total_time) * 100.0 if total_time > 0 else 0.0
    pct_fwd = (time_forward / total_time) * 100.0 if total_time > 0 else 0.0
    pct_bwd = (time_backward / total_time) * 100.0 if total_time > 0 else 0.0
    pct_opt = (time_optimizer / total_time) * 100.0 if total_time > 0 else 0.0

    # Profiler key averages summary
    print("[Task 2 Profiler] Aggregating profiler operator table...", flush=True)
    key_averages = prof.key_averages()
    profiler_summary_str = key_averages.table(sort_by="cpu_time_total", row_limit=25)
    summary_path = os.path.join(output_dir, "task2_profiler_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"=== PyTorch Profiler Summary (Device: {device}) ===\n")
        f.write(profiler_summary_str)

    # Export trace if requested
    trace_path = os.path.join(output_dir, "task2_trace.json")
    if export_trace:
        try:
            prof.export_chrome_trace(trace_path)
            print(f"[Task 2 Profiler] Chrome trace saved to {trace_path}", flush=True)
        except Exception as e:
            print(f"[Task 2 Profiler] Warning: Chrome trace export skipped ({e})", flush=True)
            trace_path = None

    # Save CSV breakdown
    breakdown_df = pd.DataFrame([
        {"Phase": "Data Loading", "Time_sec": time_data_loading, "Time_ms": time_data_loading * 1000.0, "Share_pct": pct_data},
        {"Phase": "Forward Pass", "Time_sec": time_forward, "Time_ms": time_forward * 1000.0, "Share_pct": pct_fwd},
        {"Phase": "Backward Pass", "Time_sec": time_backward, "Time_ms": time_backward * 1000.0, "Share_pct": pct_bwd},
        {"Phase": "Optimizer Step", "Time_sec": time_optimizer, "Time_ms": time_optimizer * 1000.0, "Share_pct": pct_opt},
        {"Phase": "Total Epoch", "Time_sec": total_time, "Time_ms": total_time * 1000.0, "Share_pct": 100.0},
    ])
    csv_path = os.path.join(output_dir, "task2_profiling_breakdown.csv")
    breakdown_df.to_csv(csv_path, index=False)
    print(f"[Task 2 Profiler] Profiling breakdown CSV saved to {csv_path}", flush=True)

    # Generate Visualization
    plot_path = os.path.join(plot_dir, "task2_profiling_breakdown.png")
    plot_profiling_results(breakdown_df.iloc[:4], plot_path, device_name=str(device))

    print("\n--- Task 2 Profiling Breakdown Summary ---", flush=True)
    for _, row in breakdown_df.iterrows():
        print(f"  {row['Phase']:<16}: {row['Time_sec']:6.3f}s ({row['Time_ms']:8.1f} ms) | {row['Share_pct']:5.2f}%", flush=True)

    return {
        "breakdown_df": breakdown_df,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "trace_path": trace_path,
        "plot_path": plot_path,
        "total_time_sec": total_time,
        "shares": {
            "data_loading": pct_data,
            "forward_pass": pct_fwd,
            "backward_pass": pct_bwd,
            "optimizer_step": pct_opt,
        },
    }


def plot_profiling_results(df: pd.DataFrame, save_path: str, device_name: str = "mps") -> None:
    """Creates a side-by-side donut chart and bar chart showing profiling breakdown."""
    setup_plot_style()
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

    fig, (ax_donut, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Donut Chart
    wedges, texts, autotexts = ax_donut.pie(
        df["Share_pct"],
        labels=df["Phase"],
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    plt.setp(autotexts, size=11, weight="bold")
    ax_donut.set_title(f"Share of Training Time (%)\n(Device: {device_name.upper()})", fontsize=13, weight="bold")

    # Horizontal Bar Chart
    y_pos = np.arange(len(df))
    bars = ax_bar.barh(y_pos, df["Time_ms"], color=colors, edgecolor="black", alpha=0.85)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(df["Phase"])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Time per Epoch (milliseconds)")
    ax_bar.set_title(f"Absolute Time Breakdown (ms)\nTotal: {df['Time_ms'].sum():.1f} ms", fontsize=13, weight="bold")

    # Add data labels to bars
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(
            width + (0.01 * df["Time_ms"].max()),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f} ms",
            ha="left",
            va="center",
            fontsize=10,
            weight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Task 2 Profiler] Visualization saved to {save_path}", flush=True)
