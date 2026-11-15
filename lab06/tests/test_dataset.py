"""
Unit tests for Fashion-MNIST dataset loader and determinism.
"""

import pytest
import torch
from lab06.src.dataset import get_fashion_mnist_dataloaders


def test_dataset_partitions_and_shapes():
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir="lab03/data",
        batch_size=64,
        seed=42,
        num_workers=0,
        val_split=5000,
    )

    assert len(train_loader.dataset) == 55000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000

    # Fetch one batch
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == (64, 1, 28, 28)
    assert targets.shape == (64,)
    assert targets.dtype == torch.long


def test_dataset_reproducibility():
    tl1, _, _ = get_fashion_mnist_dataloaders(
        data_dir="lab03/data",
        batch_size=32,
        seed=42,
        num_workers=0,
    )
    tl2, _, _ = get_fashion_mnist_dataloaders(
        data_dir="lab03/data",
        batch_size=32,
        seed=42,
        num_workers=0,
    )

    b1_in, b1_tar = next(iter(tl1))
    b2_in, b2_tar = next(iter(tl2))

    assert torch.equal(b1_in, b2_in), "Batches from identical seed should match exactly"
    assert torch.equal(b1_tar, b2_tar), "Targets from identical seed should match exactly"
