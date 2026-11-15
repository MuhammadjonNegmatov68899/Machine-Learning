"""
Dataset loading and pre-processing pipeline for Fashion-MNIST.
Reuses and extends the configuration from Lab 3.
"""

from pathlib import Path
from typing import Tuple, Optional
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def resolve_data_dir(preferred_path: str = "data") -> str:
    """
    Locates the Fashion-MNIST dataset across standard locations
    (e.g., Lab 3 data folder or current lab data folder) to avoid redundant downloads.
    """
    candidates = [
        Path(preferred_path),
        Path("data"),
        Path("../data"),
        Path("lab03/data"),
        Path("../lab03/data"),
        Path("../../lab03/data"),
    ]

    for candidate in candidates:
        raw_dir = candidate / "FashionMNIST" / "raw"
        if raw_dir.exists() and (raw_dir / "train-images-idx3-ubyte").exists():
            return str(candidate.resolve())

    # Fallback to preferred path
    return preferred_path


def get_fashion_mnist_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_split: int = 5000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads Fashion-MNIST, normalizes using official mean (0.2860) and std (0.3530),
    deterministically splits the official 60,000 training set into train (60,000 - val_split)
    and validation (val_split) subsets, and returns DataLoaders for train, val, and test.

    Args:
        data_dir: Base directory containing or targeted for Fashion-MNIST data.
        batch_size: Mini-batch size.
        seed: Random seed for reproducible splitting and batch shuffling.
        num_workers: Number of background subprocesses for data loading.
        pin_memory: If True, pin memory in DataLoader (useful for CUDA/MPS transfers).
        val_split: Size of validation partition (default: 5000).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    resolved_dir = resolve_data_dir(data_dir)
    os.makedirs(resolved_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train_dataset = datasets.FashionMNIST(
        root=resolved_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=resolved_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = len(full_train_dataset) - val_split
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_split],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
