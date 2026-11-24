"""
Dataset pipeline for Fashion-MNIST loading, splitting, and preprocessing.
Reuses existing data repositories across labs to prevent redundant downloads.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def resolve_data_dir(preferred_path: str = "data") -> str:
    """
    Locates the Fashion-MNIST dataset directory across standard workspace paths.
    """
    candidates = [
        Path(preferred_path),
        Path("data"),
        Path("../data"),
        Path("lab03/data"),
        Path("../lab03/data"),
        Path("../../lab03/data"),
        Path("/kaggle/working/data"),
        Path("/kaggle/working/lab07/data"),
        Path("/Users/itsnegmatov/Documents/University/Machine Learning/lab03/data"),
    ]

    for candidate in candidates:
        raw_dir = candidate / "FashionMNIST" / "raw"
        if raw_dir.exists() and (raw_dir / "train-images-idx3-ubyte").exists():
            return str(candidate.resolve())

    return preferred_path


def get_fashion_mnist_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    test_batch_size: int = 256,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test DataLoaders for Fashion-MNIST.
    Uses canonical mean and standard deviation: mean=0.2860, std=0.3530.
    """
    resolved_dir = resolve_data_dir(data_dir)

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

    # Deterministic train/validation split
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
