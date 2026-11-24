"""
Tests for structured filter pruning, tensor dimension consistency, and physical model contraction.
"""

import pytest
import torch
from lab07.src.models import BaselineCNN
from lab07.src.pruning import apply_structured_filter_pruning


@pytest.mark.parametrize("ratio", [0.10, 0.25, 0.50])
def test_structured_filter_pruning_contraction(baseline_model, ratio):
    """
    Verifies that structured pruning physically removes filters, reduces parameter count,
    and produces a valid smaller model architecture.
    """
    orig_params = baseline_model.count_parameters()
    pruned_model = apply_structured_filter_pruning(baseline_model, ratio)
    new_params = pruned_model.count_parameters()

    assert new_params < orig_params
    assert isinstance(pruned_model, BaselineCNN)

    # Check that channel counts were reduced
    assert pruned_model.c1 <= baseline_model.c1
    assert pruned_model.c2 <= baseline_model.c2
    assert pruned_model.c3 <= baseline_model.c3
    assert pruned_model.c4 <= baseline_model.c4

    # Verify forward pass on multiple batch sizes
    for b in [1, 4, 16]:
        dummy_x = torch.randn(b, 1, 28, 28)
        logits = pruned_model(dummy_x)
        assert logits.shape == (b, 10)
        assert not torch.isnan(logits).any()


def test_monotonic_parameter_reduction(baseline_model):
    """Verifies that increasing structured pruning ratio strictly reduces parameter count."""
    p10 = apply_structured_filter_pruning(baseline_model, 0.10).count_parameters()
    p25 = apply_structured_filter_pruning(baseline_model, 0.25).count_parameters()
    p50 = apply_structured_filter_pruning(baseline_model, 0.50).count_parameters()

    assert baseline_model.count_parameters() > p10 > p25 > p50


def test_structured_zero_ratio_preserves(baseline_model):
    """Verifies that ratio 0.0 returns equivalent parameter count."""
    p0 = apply_structured_filter_pruning(baseline_model, 0.0)
    assert p0.count_parameters() == baseline_model.count_parameters()
