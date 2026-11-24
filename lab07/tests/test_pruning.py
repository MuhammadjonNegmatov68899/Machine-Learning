"""
Tests for global and layer-wise unstructured magnitude pruning algorithms.
"""

import pytest
import torch
from lab07.src.pruning import (
    apply_global_unstructured_pruning,
    apply_layerwise_unstructured_pruning,
    calculate_sparsity_statistics,
)


@pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.50, 0.75, 0.90, 0.95])
def test_global_pruning_sparsity(baseline_model, sparsity):
    """Verifies that global pruning achieves the exact requested sparsity target."""
    pruned_m, masks = apply_global_unstructured_pruning(baseline_model, sparsity)
    stats = calculate_sparsity_statistics(pruned_m)

    # Actual sparsity ratio should match target within 0.1% tolerance
    actual = stats["overall_sparsity_ratio"]
    assert pytest.approx(actual, abs=0.005) == sparsity

    # Verify that masked elements are exactly zero
    for name, mask in masks.items():
        weight = dict(pruned_m.named_parameters())[name]
        zeros = weight[mask == 0.0]
        if zeros.numel() > 0:
            assert torch.all(zeros == 0.0)


@pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.50, 0.75, 0.90, 0.95])
def test_layerwise_pruning_sparsity(baseline_model, sparsity):
    """Verifies that layer-wise pruning achieves the requested sparsity in each layer."""
    pruned_m, masks = apply_layerwise_unstructured_pruning(baseline_model, sparsity)
    stats = calculate_sparsity_statistics(pruned_m)

    for layer_name, layer_info in stats["layers"].items():
        layer_sparsity = layer_info["sparsity_ratio"]
        assert pytest.approx(layer_sparsity, abs=0.01) == sparsity

    actual_overall = stats["overall_sparsity_ratio"]
    assert pytest.approx(actual_overall, abs=0.005) == sparsity


def test_zero_sparsity_identity(baseline_model):
    """Verifies that 0.0 sparsity leaves all weights identical."""
    pruned_m, masks = apply_global_unstructured_pruning(baseline_model, 0.0)
    for name, p_orig in baseline_model.named_parameters():
        p_pruned = dict(pruned_m.named_parameters())[name]
        assert torch.equal(p_orig, p_pruned)
