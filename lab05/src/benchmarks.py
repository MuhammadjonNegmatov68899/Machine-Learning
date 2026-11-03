"""
Framework Performance, Computation Graph Inspection, and Benchmarks.

Implements Task 6:
- Inspection of PyTorch computation graphs (autograd grad_fn chain and TorchScript IR).
- Benchmarking eager vs traced (torch.jit.trace), scripted, and compiled execution.
- Tensor shape, dtype, strides, and memory contiguity analysis.
- Empirical benchmarking of operations on contiguous vs non-contiguous layouts.
- Plotting comprehensive benchmark figures.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from src.neural_network import PyTorchMLP
except ImportError:
    from .neural_network import PyTorchMLP


# =============================================================================
# 1. Computation Graph Inspection
# =============================================================================

def inspect_pytorch_graph(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Inspect the computation graph of a PyTorch model.

    Walks the backward graph via grad_fn and traces the forward graph
    via torch.jit.trace to capture the intermediate representation (IR).

    Returns:
        Dictionary containing node descriptions, backward graph trace,
        and TorchScript intermediate representation.
    """
    model.eval()

    # 1. Trace autograd backward graph
    x = sample_input.clone().detach().requires_grad_(True)
    out = model(x)
    loss = out.sum()

    backward_nodes: List[str] = []
    visited: set = set()

    def walk_grad_fn(grad_fn: Any, depth: int = 0) -> None:
        if grad_fn is None or grad_fn in visited:
            return
        visited.add(grad_fn)
        fn_name = type(grad_fn).__name__
        backward_nodes.append(f"{'  ' * depth}├── {fn_name}")
        for next_fn, _ in getattr(grad_fn, "next_functions", []):
            walk_grad_fn(next_fn, depth + 1)

    walk_grad_fn(loss.grad_fn)

    # 2. Trace forward TorchScript IR
    traced_model = torch.jit.trace(model, sample_input)
    torchscript_graph_str = str(traced_model.graph)
    torchscript_code_str = str(traced_model.code)

    return {
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "backward_nodes": backward_nodes,
        "backward_graph_text": "\n".join(backward_nodes),
        "torchscript_ir": torchscript_graph_str,
        "torchscript_code": torchscript_code_str,
    }


# =============================================================================
# 2. Execution Benchmarks (Eager vs Traced vs Compiled)
# =============================================================================

def run_execution_benchmarks(
    batch_size: int = 32,
    input_dim: int = 2,
    hidden_dim: int = 8,
    output_dim: int = 1,
    warmup_iters: int = 100,
    timed_iters: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Benchmark inference execution across PyTorch execution modes:
    1. Eager execution (standard PyTorch forward).
    2. Traced execution (torch.jit.trace).
    3. Scripted execution (torch.jit.script).
    4. Compiled execution (torch.compile with aot_eager backend).
    5. Compiled execution with Inductor backend (attempted, failure logged).

    Warmup iterations precede timing to avoid initialization/compilation artifacts.
    """
    torch.manual_seed(seed)
    model = PyTorchMLP(
        nin=input_dim,
        hidden_dim=hidden_dim,
        nout=output_dim,
        hidden_activation="sigmoid",
        seed=seed,
    ).eval()

    sample_input = torch.randn(batch_size, input_dim, dtype=torch.float32)

    modes: Dict[str, Any] = {}

    # 1. Eager
    modes["eager"] = model

    # 2. Traced (TorchScript)
    try:
        modes["traced (torch.jit.trace)"] = torch.jit.trace(model, sample_input)
    except Exception as e:
        modes["traced (torch.jit.trace)"] = f"ERROR: {e}"

    # 3. Scripted
    try:
        modes["scripted (torch.jit.script)"] = torch.jit.script(model)
    except Exception as e:
        modes["scripted (torch.jit.script)"] = f"ERROR: {e}"

    # 4. Compiled (aot_eager)
    try:
        modes["compiled (aot_eager)"] = torch.compile(model, backend="aot_eager")
    except Exception as e:
        modes["compiled (aot_eager)"] = f"ERROR: {e}"

    # 5. Compiled (inductor)
    try:
        compiled_ind = torch.compile(model, backend="inductor")
        # Trigger compilation in warmup
        compiled_ind(sample_input)
        modes["compiled (inductor)"] = compiled_ind
    except Exception as e:
        # Preserve actual exception type and full message without pre-determined speculation
        modes["compiled (inductor)"] = f"{type(e).__name__}: {str(e).strip()}"

    records: List[Dict[str, Any]] = []
    eager_median_latency = None

    for mode_name, fn in modes.items():
        if isinstance(fn, str):
            # Error or unsupported: capture exact exception text and type
            records.append({
                "mode": mode_name,
                "status": "UNSUPPORTED",
                "warmup_iters": warmup_iters,
                "timed_iters": timed_iters,
                "mean_latency_us": np.nan,
                "median_latency_us": np.nan,
                "std_latency_us": np.nan,
                "min_latency_us": np.nan,
                "max_latency_us": np.nan,
                "throughput_inferences_per_sec": np.nan,
                "speedup_vs_eager": np.nan,
                "notes": " ".join(fn.split()),
            })
            continue

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = fn(sample_input)

        # Timed iterations
        latencies_sec: List[float] = []
        with torch.no_grad():
            for _ in range(timed_iters):
                t0 = time.perf_counter()
                _ = fn(sample_input)
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)

        lat_us = np.array(latencies_sec) * 1e6
        mean_us = float(np.mean(lat_us))
        median_us = float(np.median(lat_us))
        std_us = float(np.std(lat_us))
        min_us = float(np.min(lat_us))
        max_us = float(np.max(lat_us))
        throughput = float(batch_size * timed_iters / np.sum(latencies_sec))

        if mode_name == "eager":
            eager_median_latency = median_us
            speedup = 1.0
        else:
            speedup = (
                float(eager_median_latency / median_us)
                if eager_median_latency
                else np.nan
            )

        records.append({
            "mode": mode_name,
            "status": "SUCCESS",
            "warmup_iters": warmup_iters,
            "timed_iters": timed_iters,
            "mean_latency_us": mean_us,
            "median_latency_us": median_us,
            "std_latency_us": std_us,
            "min_latency_us": min_us,
            "max_latency_us": max_us,
            "throughput_inferences_per_sec": throughput,
            "speedup_vs_eager": speedup,
            "notes": "Optimal execution" if speedup >= 1.0 else "Framework dispatch overhead dominates",
        })

    return pd.DataFrame(records)


# =============================================================================
# 3. Tensor Contiguity and Memory Layout Benchmarks
# =============================================================================

def inspect_tensor_layout(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
    """Inspect dimensional and memory layout properties of a PyTorch tensor."""
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "element_size_bytes": tensor.element_size(),
        "total_elements": tensor.numel(),
        "total_memory_kb": (tensor.numel() * tensor.element_size()) / 1024.0,
        "is_contiguous": tensor.is_contiguous(),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
    }


def run_tensor_layout_benchmarks(
    rows: int = 2048,
    cols: int = 2048,
    warmup_iters: int = 50,
    timed_iters: int = 500,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Benchmark operations across contiguous and non-contiguous tensor layouts.

    Tests:
    1. Row-wise reduction: sum(dim=1)
    2. Column-wise reduction: sum(dim=0)
    3. Elementwise addition: x + 1.0
    4. Reshape / Flattening: contiguous view vs non-contiguous copy requirement

    Examines the impact of cache locality and stride jumps on CPU memory hierarchy.
    """
    torch.manual_seed(seed)

    # 1. Contiguous base tensor (Row-Major, stride = [cols, 1])
    x_contig = torch.randn(rows, cols, dtype=dtype)

    # 2. Non-contiguous transposed view (Column-Major view, stride = [1, cols])
    x_noncontig = x_contig.t()

    info_contig = inspect_tensor_layout(x_contig, "contiguous_matrix")
    info_noncontig = inspect_tensor_layout(x_noncontig, "transposed_view")

    operations = [
        ("Row-wise Reduction (sum dim=1)", lambda t: t.sum(dim=1)),
        ("Column-wise Reduction (sum dim=0)", lambda t: t.sum(dim=0)),
        ("Elementwise Addition (t + 1.0)", lambda t: t + 1.0),
        ("Elementwise Multiplication (t * 2.0)", lambda t: t * 2.0),
    ]

    records: List[Dict[str, Any]] = []

    for op_name, op_fn in operations:
        # Warmup contiguous
        for _ in range(warmup_iters):
            _ = op_fn(x_contig)

        # Timed contiguous
        t_contig_list: List[float] = []
        for _ in range(timed_iters):
            t0 = time.perf_counter()
            _ = op_fn(x_contig)
            t1 = time.perf_counter()
            t_contig_list.append(t1 - t0)

        # Warmup non-contiguous
        for _ in range(warmup_iters):
            _ = op_fn(x_noncontig)

        # Timed non-contiguous
        t_noncontig_list: List[float] = []
        for _ in range(timed_iters):
            t0 = time.perf_counter()
            _ = op_fn(x_noncontig)
            t1 = time.perf_counter()
            t_noncontig_list.append(t1 - t0)

        c_mean_ms = float(np.mean(t_contig_list) * 1e3)
        c_median_ms = float(np.median(t_contig_list) * 1e3)
        nc_mean_ms = float(np.mean(t_noncontig_list) * 1e3)
        nc_median_ms = float(np.median(t_noncontig_list) * 1e3)

        ratio = nc_median_ms / c_median_ms if c_median_ms > 0 else np.nan

        # Bandwidth estimation: memory read/written
        bytes_transferred = (rows * cols * x_contig.element_size())
        c_bandwidth_gb_s = (bytes_transferred / (np.median(t_contig_list) * 1e9))
        nc_bandwidth_gb_s = (bytes_transferred / (np.median(t_noncontig_list) * 1e9))

        records.append({
            "operation": op_name,
            "tensor_shape": f"[{rows}, {cols}]",
            "dtype": str(dtype),
            "contiguous_stride": str(info_contig["stride"]),
            "noncontiguous_stride": str(info_noncontig["stride"]),
            "contiguous_median_ms": c_median_ms,
            "noncontiguous_median_ms": nc_median_ms,
            "slowdown_factor": ratio,
            "contiguous_bandwidth_GB_s": c_bandwidth_gb_s,
            "noncontiguous_bandwidth_GB_s": nc_bandwidth_gb_s,
        })

    df = pd.DataFrame(records)
    layout_meta = {
        "contiguous": info_contig,
        "noncontiguous": info_noncontig,
    }
    return df, layout_meta


# =============================================================================
# 4. System Environment Metadata
# =============================================================================

def get_system_metadata() -> Dict[str, str]:
    """Capture exact hardware and software environment details."""
    return {
        "os_platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_available": str(torch.backends.mps.is_available()),
        "cuda_available": str(torch.cuda.is_available()),
        "torch_num_threads": str(torch.get_num_threads()),
    }


# =============================================================================
# 5. Visualization Generation
# =============================================================================

def generate_benchmark_plots(
    training_plot_data: Dict[str, Any],
    exec_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate professional multi-panel figure summarizing:
    1. Training loss curves: Custom Engine vs PyTorch.
    2. Multi-run training time comparison (5 seeds + median).
    3. PyTorch execution modes latency comparison.
    4. Contiguous vs non-contiguous memory layout slowdown.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # ---------------------------------------------------------
    # Panel 1: Training Loss Curves
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    epochs = range(1, len(training_plot_data["custom_loss_histories"][0]) + 1)
    # Plot seed 42 (primary seed) and shaded band across seeds
    custom_arr = np.array(training_plot_data["custom_loss_histories"])
    torch_arr = np.array(training_plot_data["torch_loss_histories"])

    ax1.plot(
        epochs,
        np.median(custom_arr, axis=0),
        color="#1f77b4",
        lw=2.5,
        label="Custom Engine (Median)",
    )
    ax1.fill_between(
        epochs,
        np.min(custom_arr, axis=0),
        np.max(custom_arr, axis=0),
        color="#1f77b4",
        alpha=0.15,
    )

    ax1.plot(
        epochs,
        np.median(torch_arr, axis=0),
        color="#ff7f0e",
        lw=2.5,
        linestyle="--",
        label="PyTorch (Median)",
    )
    ax1.fill_between(
        epochs,
        np.min(torch_arr, axis=0),
        np.max(torch_arr, axis=0),
        color="#ff7f0e",
        alpha=0.15,
    )

    ax1.set_title("A. Training Loss Convergence (5-Seed Envelope)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # ---------------------------------------------------------
    # Panel 2: Multi-Seed Training Duration Comparison
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    seeds = [str(s) for s in training_plot_data["seeds"]]
    custom_times = training_plot_data["custom_times"]
    torch_times = training_plot_data["torch_times"]

    x = np.arange(len(seeds) + 1)
    width = 0.35

    all_custom = custom_times + [float(np.median(custom_times))]
    all_torch = torch_times + [float(np.median(torch_times))]
    labels = seeds + ["MEDIAN"]

    ax2.bar(x - width / 2, all_custom, width, label="Custom Engine", color="#1f77b4")
    ax2.bar(x + width / 2, all_torch, width, label="PyTorch", color="#ff7f0e")

    ax2.set_title("B. Training Latency per Seed (100 Epochs)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_xlabel("Random Seed", fontsize=11)
    ax2.set_ylabel("Elapsed Time (seconds)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True)

    # ---------------------------------------------------------
    # Panel 3: PyTorch Execution Modes Latency
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    valid_exec = exec_df[exec_df["status"] == "SUCCESS"].copy()
    modes = valid_exec["mode"].tolist()
    medians = valid_exec["median_latency_us"].tolist()
    colors = ["#2ca02c" if m == "eager" else "#9467bd" for m in modes]

    bars = ax3.barh(modes, medians, color=colors, height=0.55)
    ax3.set_title("C. PyTorch Inference Latency (Batch=32)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Median Latency (microseconds)", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Annotate bars
    for bar in bars:
        val = bar.get_width()
        ax3.text(
            val + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} µs",
            va="center",
            fontsize=10,
            fontweight="semibold",
        )

    # ---------------------------------------------------------
    # Panel 4: Memory Layout Impact (Contiguous vs Non-Contiguous)
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    ops = [op.split("(")[0].strip() for op in layout_df["operation"]]
    c_lat = layout_df["contiguous_median_ms"].tolist()
    nc_lat = layout_df["noncontiguous_median_ms"].tolist()

    y = np.arange(len(ops))
    height = 0.35

    ax4.barh(y - height / 2, c_lat, height, label="Contiguous (Row-Major)", color="#2ca02c")
    ax4.barh(y + height / 2, nc_lat, height, label="Non-Contiguous (Transposed)", color="#d62728")

    ax4.set_title("D. Memory Layout Overhead (2048x2048 Matrix)", fontsize=12, fontweight="bold")
    ax4.set_yticks(y)
    ax4.set_yticklabels(ops)
    ax4.set_xlabel("Median Latency (milliseconds)", fontsize=11)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
