"""
Tests for checkpoint saving, loading, and restoration fidelity.
"""

import os
import tempfile
import pytest
import torch
from lab07.src.models import BaselineCNN, StudentCNN


def test_baseline_checkpoint_roundtrip(baseline_model, dummy_input):
    """Verifies that saving and loading a BaselineCNN restores identical state and outputs."""
    orig_output = baseline_model(dummy_input)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": baseline_model.state_dict(),
                "val_acc": 0.895,
            },
            ckpt_path,
        )

        loaded_model = BaselineCNN()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_output = loaded_model(dummy_input)

        assert torch.allclose(orig_output, loaded_output, atol=1e-6)
        assert ckpt["epoch"] == 5
        assert ckpt["val_acc"] == 0.895
    finally:
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)


def test_student_checkpoint_roundtrip(student_model, dummy_input):
    """Verifies that saving and loading a StudentCNN restores identical state and outputs."""
    orig_output = student_model(dummy_input)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        torch.save(
            {
                "epoch": 10,
                "student_state_dict": student_model.state_dict(),
                "val_acc": 0.880,
            },
            ckpt_path,
        )

        loaded_student = StudentCNN()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        loaded_student.load_state_dict(ckpt["student_state_dict"])
        loaded_output = loaded_student(dummy_input)

        assert torch.allclose(orig_output, loaded_output, atol=1e-6)
        assert ckpt["epoch"] == 10
    finally:
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
