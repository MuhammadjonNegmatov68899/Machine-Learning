"""
Benchmarking suite for Lab 07:
1. Precise inference latency & throughput profiling on CPU and MPS.
2. Dense vs sparse disk storage footprint measurement.
3. FLOPs, MACs, parameter counts, and theoretical compute metrics.
"""

from typing import Dict, Any, Tuple, Optional, List
import io
import os
import gzip
import time
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lab07.src.utils import count_parameters, compute_macs_and_flops


def benchmark_latency(
    model: nn.Module,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    warmup_runs: int = 20,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Measures inference latency with high-precision timers and proper synchronization.
    
    Args:
        model: Evaluated PyTorch model.
        batch_size: Inference batch size.
        device: Device to profile on (CPU or MPS).
        warmup_runs: Number of unmeasured warm-up iterations.
        num_runs: Number of timed iterations.
        
    Returns:
        Dictionary containing mean, median, std, min, max, p95, p99 latency in ms,
        and throughput in samples/sec.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)

    is_mps = device.type == "mps"
    is_cuda = device.type == "cuda"

    def sync():
        if is_mps:
            torch.mps.synchronize()
        elif is_cuda:
            torch.cuda.synchronize()

    # Warmup phase
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            sync()

    # Measurement phase
    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            sync()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            sync()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

    latencies_arr = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies_arr))
    median_ms = float(np.median(latencies_arr))
    std_ms = float(np.std(latencies_arr))
    min_ms = float(np.min(latencies_arr))
    max_ms = float(np.max(latencies_arr))
    p95_ms = float(np.percentile(latencies_arr, 95))
    p99_ms = float(np.percentile(latencies_arr, 99))

    throughput_samples_per_sec = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_ms, 3),
        "median_latency_ms": round(median_ms, 3),
        "std_latency_ms": round(std_ms, 3),
        "min_latency_ms": round(min_ms, 3),
        "max_latency_ms": round(max_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "p99_latency_ms": round(p99_ms, 3),
        "throughput_samples_sec": round(throughput_samples_per_sec, 2),
        "batch_size": batch_size,
    }


def measure_storage_footprint(model: nn.Module) -> Dict[str, Any]:
    """
    Measures on-disk model size across:
    1. Dense uncompressed state_dict (.pt format).
    2. Dense compressed format (.pt.gz using gzip/deflate).
    3. Sparse COO representation (.pt sparse tensor dictionary).
    
    Returns:
        Dictionary of sizes in bytes and megabytes (MB) / kilobytes (KB).
    """
    model_cpu = model.to("cpu")
    state_dict = model_cpu.state_dict()

    # 1. Dense uncompressed .pt size
    buf_dense = io.BytesIO()
    torch.save(state_dict, buf_dense)
    dense_bytes = len(buf_dense.getvalue())

    # 2. Dense compressed .pt.gz size
    buf_dense.seek(0)
    gz_bytes = len(gzip.compress(buf_dense.getvalue()))

    # 3. Sparse COO representation
    sparse_state_dict = {}
    for k, v in state_dict.items():
        if "weight" in k and v.dim() >= 2:
            sparse_state_dict[k] = v.to_sparse()
        else:
            sparse_state_dict[k] = v

    buf_sparse = io.BytesIO()
    torch.save(sparse_state_dict, buf_sparse)
    sparse_coo_bytes = len(buf_sparse.getvalue())

    return {
        "dense_size_kb": round(dense_bytes / 1024.0, 2),
        "dense_size_mb": round(dense_bytes / (1024.0 * 1024.0), 4),
        "compressed_size_kb": round(gz_bytes / 1024.0, 2),
        "compressed_size_mb": round(gz_bytes / (1024.0 * 1024.0), 4),
        "sparse_coo_size_kb": round(sparse_coo_bytes / 1024.0, 2),
        "sparse_coo_size_mb": round(sparse_coo_bytes / (1024.0 * 1024.0), 4),
    }


def profile_full_model_metrics(
    model: nn.Module,
    name: str,
    device: torch.device,
    cpu_latency: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive profiling for a model across parameters, sparsity, FLOPs, storage, and latency.
    """
    param_info = count_parameters(model)
    storage_info = measure_storage_footprint(model)
    compute_info = compute_macs_and_flops(model, input_size=(1, 1, 28, 28))

    # CPU Latency (consistent baseline for dense kernel analysis)
    cpu_bench = benchmark_latency(
        model.to("cpu"), batch_size=64, device=torch.device("cpu"), warmup_runs=15, num_runs=50
    )

    # Device Latency (e.g. MPS if available)
    if device.type != "cpu":
        device_bench = benchmark_latency(
            model.to(device), batch_size=64, device=device, warmup_runs=15, num_runs=50
        )
    else:
        device_bench = cpu_bench

    return {
        "model_name": name,
        **param_info,
        **storage_info,
        **compute_info,
        "cpu_median_latency_ms": cpu_bench["median_latency_ms"],
        "cpu_mean_latency_ms": cpu_bench["mean_latency_ms"],
        "cpu_throughput": cpu_bench["throughput_samples_sec"],
        "device_median_latency_ms": device_bench["median_latency_ms"],
        "device_mean_latency_ms": device_bench["mean_latency_ms"],
        "device_throughput": device_bench["throughput_samples_sec"],
    }
