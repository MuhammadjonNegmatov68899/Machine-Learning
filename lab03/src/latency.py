"""
Inference latency benchmarking module following rigorous measurement protocol.
"""

from typing import Dict, Any, List
import time
import torch
import torch.nn as nn
import numpy as np


def synchronize(device: torch.device) -> None:
    """Synchronizes GPU / accelerator execution queue if applicable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_latency(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    num_runs: int,
    warmup_runs: int = 10,
    input_shape: tuple = (1, 28, 28),
) -> Dict[str, Any]:
    """
    Measures inference latency for a given model and batch size.

    Args:
        model: PyTorch model to benchmark.
        batch_size: Mini-batch size (e.g. 1 or 128).
        device: Device on which benchmarking is conducted ('cpu', 'mps', or 'cuda').
        num_runs: Number of timed evaluation runs (100 for batch 1, 20 for batch 128).
        warmup_runs: Number of untimed warmup iterations.
        input_shape: Spatial shape of single sample (C, H, W).

    Returns:
        Dict with median, mean, min, max, std latencies in milliseconds,
        throughput in samples/sec, and latency per sample.
    """
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Warm-up phase
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            synchronize(device)

    # Timed runs
    latencies_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(num_runs):
            synchronize(device)
            t_start = time.perf_counter()
            _ = model(dummy_input)
            synchronize(device)
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    median_latency = float(np.median(latencies_arr))
    mean_latency = float(np.mean(latencies_arr))
    min_latency = float(np.min(latencies_arr))
    max_latency = float(np.max(latencies_arr))
    std_latency = float(np.std(latencies_arr))
    throughput = float(batch_size / (median_latency / 1000.0))  # samples / sec
    per_sample_latency = float(median_latency / batch_size)

    return {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "device": str(device),
        "median_ms": median_latency,
        "mean_ms": mean_latency,
        "min_ms": min_latency,
        "max_ms": max_latency,
        "std_ms": std_latency,
        "throughput_samples_per_sec": throughput,
        "per_sample_ms": per_sample_latency,
        "raw_latencies": latencies_ms,
    }


def benchmark_model_latencies(
    model: nn.Module,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmarks model latency for batch 1 (100 runs) and batch 128 (20 runs),
    as required by Lab Manual Section 3.
    """
    res_b1 = benchmark_latency(model, batch_size=1, device=device, num_runs=100, warmup_runs=15)
    res_b128 = benchmark_latency(model, batch_size=128, device=device, num_runs=20, warmup_runs=10)
    return {
        "batch_1": res_b1,
        "batch_128": res_b128,
    }
