"""
Utility functions for random seeding, hardware resource monitoring,
configuration management, and plot styling.
"""

from typing import Dict, Any, Optional
import os
import random
import yaml
import numpy as np
import psutil
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python random, NumPy, and PyTorch for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS random seed is set via torch.manual_seed
        pass


def get_device(preference: str = "auto") -> torch.device:
    """
    Selects target computational device based on preference and hardware availability.
    """
    pref = preference.lower().strip()
    if pref == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif pref in ("mps", "cuda", "cpu"):
        if pref == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("cpu")
        if pref == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(pref)
    return torch.device("cpu")


def get_memory_info() -> Dict[str, float]:
    """
    Returns current process RSS memory in MB, and device-specific memory if available.
    """
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
    info = {"rss_mb": float(rss_mb)}

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            mps_alloc_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
            info["mps_allocated_mb"] = float(mps_alloc_mb)
        except Exception:
            pass
    elif torch.cuda.is_available():
        try:
            cuda_alloc_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            info["cuda_allocated_mb"] = float(cuda_alloc_mb)
        except Exception:
            pass

    return info


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file into a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML file at {path} to parse into a dictionary.")
    return config


def save_yaml_config(config: Dict[str, Any], path: str) -> None:
    """Saves a Python dictionary to a YAML configuration file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flattens a nested dictionary into a single-level dictionary with dotted keys."""
    items = []
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def setup_plot_style() -> None:
    """Configures clean, consistent publication styling for matplotlib/seaborn plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": True,
    })
