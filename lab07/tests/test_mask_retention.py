"""
Tests verifying that pruning masks remain strictly fixed and zeroed during optimization.
"""

import pytest
import torch
import torch.nn as nn
from lab07.src.pruning import (
    apply_global_unstructured_pruning,
    zero_masked_gradients,
    apply_pruning_masks,
    verify_mask_adherence,
)


def test_mask_retention_under_adam_optimizer(baseline_model):
    """
    Simulates training steps with Adam (including momentum and second-moment updates)
    and verifies that masked parameters remain strictly zero after every update.
    """
    pruned_m, masks = apply_global_unstructured_pruning(baseline_model, 0.50)
    optimizer = torch.optim.Adam(pruned_m.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Record non-masked weight copy to ensure unmasked weights are actually updated
    conv1_orig_unmasked = pruned_m.conv1.weight.data[masks["conv1.weight"] == 1.0].clone()

    for step in range(5):
        optimizer.zero_grad()
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))
        out = pruned_m(x)
        loss = criterion(out, y)
        loss.backward()

        zero_masked_gradients(pruned_m, masks)
        optimizer.step()
        apply_pruning_masks(pruned_m, masks)

        assert verify_mask_adherence(pruned_m, masks, tolerance=0.0)

    # Ensure unmasked weights actually learned / changed
    conv1_new_unmasked = pruned_m.conv1.weight.data[masks["conv1.weight"] == 1.0]
    diff = (conv1_new_unmasked - conv1_orig_unmasked).abs().max().item()
    assert diff > 1e-5, "Unmasked weights were unexpectedly frozen!"


def test_mask_retention_under_sgd_momentum(baseline_model):
    """
    Verifies mask retention under SGD with high momentum and weight decay.
    """
    pruned_m, masks = apply_global_unstructured_pruning(baseline_model, 0.75)
    optimizer = torch.optim.SGD(
        pruned_m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):
        optimizer.zero_grad()
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))
        loss = criterion(pruned_m(x), y)
        loss.backward()

        zero_masked_gradients(pruned_m, masks)
        optimizer.step()
        apply_pruning_masks(pruned_m, masks)

        assert verify_mask_adherence(pruned_m, masks, tolerance=0.0)
