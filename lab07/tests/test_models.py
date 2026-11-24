"""
Tests for BaselineCNN and StudentCNN architectures and parameter counts.
"""

import pytest
import torch
from lab07.src.models import BaselineCNN, StudentCNN, build_baseline_model, build_student_model


def test_baseline_parameter_count(baseline_model):
    """Verifies that BaselineCNN satisfies the ~1M parameters requirement."""
    params = baseline_model.count_parameters()
    # Expect ~1,000,000 parameters (tolerance 900k to 1.1M)
    assert 900_000 <= params <= 1_100_000, f"Expected ~1M params, got {params}"
    assert params == 1_011_866, f"Exact baseline param count altered: {params}"


def test_student_parameter_count(student_model):
    """Verifies that StudentCNN satisfies the ~100k parameters requirement."""
    params = student_model.count_parameters()
    # Expect ~100,000 parameters (tolerance 90k to 110k)
    assert 90_000 <= params <= 110_000, f"Expected ~100k params, got {params}"
    assert params == 101_578, f"Exact student param count altered: {params}"


def test_baseline_forward(baseline_model, dummy_input):
    """Verifies baseline forward pass and output shape."""
    out = baseline_model(dummy_input)
    assert out.shape == (4, 10)
    assert not torch.isnan(out).any()


def test_student_forward(student_model, dummy_input):
    """Verifies student forward pass and output shape."""
    out = student_model(dummy_input)
    assert out.shape == (4, 10)
    assert not torch.isnan(out).any()


def test_factory_builders():
    """Verifies factory helper functions with default configs."""
    b = build_baseline_model()
    s = build_student_model()
    assert isinstance(b, BaselineCNN)
    assert isinstance(s, StudentCNN)
    assert b.count_parameters() == 1_011_866
    assert s.count_parameters() == 101_578


def test_prunable_weights_list(baseline_model):
    """Verifies that eligible weights are Conv2d and Linear weight tensors only."""
    prunables = baseline_model.get_prunable_weights()
    assert len(prunables) == 6  # conv1, conv2, conv3, conv4, fc1, fc2
    names = [name for name, _ in prunables]
    assert "conv1.weight" in names
    assert "conv4.weight" in names
    assert "fc1.weight" in names
    assert "fc2.weight" in names
    for name, p in prunables:
        assert "bias" not in name
