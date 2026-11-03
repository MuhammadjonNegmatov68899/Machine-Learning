"""
Shared Test Fixtures and Configurations for Lab 05.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

# Ensure lab05 root is in sys.path
LAB_DIR = Path(__file__).resolve().parent.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from src.engine import Value
from src.neural_network import MLP, PyTorchMLP


@pytest.fixture
def scalar_values():
    """Sample scalar Value nodes."""
    return Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c")


@pytest.fixture
def sample_data_moons():
    """Reproducible make_moons data."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=50, noise=0.2, random_state=42)
    return X, y


@pytest.fixture
def custom_and_torch_mlp():
    """Matching custom and PyTorch MLPs with synced weights."""
    custom_net = MLP(nin=2, nouts=[8, 1], activations=["sigmoid", "sigmoid"], seed=42)
    torch_net = PyTorchMLP(nin=2, hidden_dim=8, nout=1, hidden_activation="sigmoid", seed=42)
    torch_net.sync_weights_from_custom(custom_net)
    return custom_net, torch_net
