"""
Utility functions for reproducibility, hardware profiling, parameter counting, and metrics.
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """
    Selects the best available computational device.
    Prioritizes MPS on Apple Silicon, CUDA on Linux/Windows, fallback to CPU.
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Counts total, trainable, and non-zero parameters in a PyTorch module.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero = sum(torch.count_nonzero(p).item() for p in model.parameters())
    zero = total - nonzero
    sparsity = (zero / total * 100.0) if total > 0 else 0.0

    return {
        "total_params": total,
        "trainable_params": trainable,
        "nonzero_params": nonzero,
        "zero_params": zero,
        "sparsity_percent": round(sparsity, 4),
    }


def measure_model_size_mb(model: nn.Module) -> float:
    """
    Calculates the memory size of model parameters and buffers in megabytes (MB).
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return round(size_mb, 4)


def get_file_size_mb(filepath: str) -> float:
    """Returns the size of a file on disk in megabytes (MB)."""
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 4)
    return 0.0


def compute_macs_and_flops(
    model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 28, 28)
) -> Dict[str, Any]:
    """
    Computes Multiply-Accumulate Operations (MACs) and FLOPs.
    Uses 'thop' on a deep-copied CPU clone to prevent in-place float64 buffer injection,
    otherwise calculates analytically for CNN models.
    """
    import copy

    # Deepcopy to CPU so thop's internal float64 buffers do not corrupt the original model on MPS
    model_clone = copy.deepcopy(model).to("cpu")
    model_clone.eval()
    dummy_input = torch.randn(*input_size, device="cpu")

    try:
        from thop import profile
        macs, params = profile(model_clone, inputs=(dummy_input,), verbose=False)
        flops = 2 * macs
        return {
            "macs": int(macs),
            "flops": int(flops),
            "macs_m": round(macs / 1e6, 3),
            "flops_m": round(flops / 1e6, 3),
            "method": "thop",
        }
    except Exception:
        # Analytical fallback
        total_macs = 0
        h, w = input_size[2], input_size[3]
        in_c = input_size[1]

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                out_c = m.out_channels
                kh, kw = m.kernel_size
                stride = m.stride[0]
                pad = m.padding[0]
                out_h = (h + 2 * pad - kh) // stride + 1
                out_w = (w + 2 * pad - kw) // stride + 1
                macs_layer = out_h * out_w * out_c * (in_c * kh * kw)
                total_macs += macs_layer
                h, w, in_c = out_h, out_w, out_c
            elif isinstance(m, nn.MaxPool2d):
                kh = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                stride = m.stride if isinstance(m.stride, int) else m.stride[0]
                h = (h - kh) // stride + 1
                w = (w - kh) // stride + 1
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                output_size = m.output_size
                if isinstance(output_size, int):
                    h, w = output_size, output_size
                else:
                    h, w = output_size
            elif isinstance(m, nn.Linear):
                macs_layer = m.in_features * m.out_features
                total_macs += macs_layer

        flops = 2 * total_macs
        return {
            "macs": int(total_macs),
            "flops": int(flops),
            "macs_m": round(total_macs / 1e6, 3),
            "flops_m": round(flops / 1e6, 3),
            "method": "analytical",
        }
