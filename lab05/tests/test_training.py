"""
Unit Tests for Task 5: Neural Network Modules and make_moons Training.
"""

import math
import numpy as np
import pytest
import torch

from src.neural_network import Neuron, Layer, MLP, PyTorchMLP
from src.operations import binary_cross_entropy
from src.training import (
    compute_accuracy,
    prepare_moons_data,
    train_custom_engine,
    train_pytorch_engine,
)


class TestNeuralNetworkModules:
    """Tests for Neuron, Layer, and MLP architectures and parameters."""

    def test_neuron_parameter_count(self):
        n = Neuron(nin=3, nonlin="sigmoid")
        assert len(n.parameters()) == 4  # 3 weights + 1 bias
        assert n.nonlin == "sigmoid"

    def test_layer_parameter_count(self):
        layer = Layer(nin=2, nout=8, nonlin="sigmoid")
        assert len(layer.neurons) == 8
        # 8 neurons * (2 weights + 1 bias) = 24 parameters
        assert len(layer.parameters()) == 24

    def test_mlp_2_8_1_architecture(self):
        mlp = MLP(nin=2, nouts=[8, 1], activations=["sigmoid", "sigmoid"], seed=42)
        assert len(mlp.layers) == 2
        # Layer 1: 2 -> 8 (24 params)
        # Layer 2: 8 -> 1 (9 params: 8 weights + 1 bias)
        # Total = 33 parameters
        assert len(mlp.parameters()) == 33

    def test_pytorch_mlp_parameter_count(self):
        torch_net = PyTorchMLP(nin=2, hidden_dim=8, nout=1, hidden_activation="sigmoid", seed=42)
        total_params = sum(p.numel() for p in torch_net.parameters())
        assert total_params == 33


class TestEngineParityAndTraining:
    """Tests for Custom Engine vs PyTorch numerical parity and training dynamics."""

    def test_forward_parity_with_synced_weights(self):
        """
        Verify that Custom MLP and PyTorch MLP produce identical predictions
        when initialized with identical weights.
        """
        custom_net = MLP(nin=2, nouts=[8, 1], activations=["sigmoid", "sigmoid"], seed=42)
        torch_net = PyTorchMLP(nin=2, hidden_dim=8, nout=1, hidden_activation="sigmoid", seed=42)
        torch_net.sync_weights_from_custom(custom_net)

        test_inputs = [
            [0.5, -1.2],
            [-0.3, 0.8],
            [1.0, 1.0],
            [-1.5, -0.5],
        ]

        for x in test_inputs:
            out_c = custom_net(x).data
            out_t = torch_net(torch.tensor([x], dtype=torch.float32)).item()
            assert math.isclose(out_c, out_t, abs_tol=1e-5), (
                f"Forward mismatch: Custom={out_c}, PyTorch={out_t}"
            )

    def test_backward_parity_with_synced_weights(self):
        """
        Verify that backpropagation produces matching parameter gradients
        between Custom Engine and PyTorch.
        """
        custom_net = MLP(nin=2, nouts=[8, 1], activations=["sigmoid", "sigmoid"], seed=42)
        torch_net = PyTorchMLP(nin=2, hidden_dim=8, nout=1, hidden_activation="sigmoid", seed=42)
        torch_net.sync_weights_from_custom(custom_net)

        x = [0.8, -0.5]
        target = 1.0

        # Custom forward and backward
        custom_net.zero_grad()
        pred_c = custom_net(x)
        loss_c = binary_cross_entropy([pred_c], [target])
        loss_c.backward()

        # PyTorch forward and backward
        torch_net.zero_grad()
        xt = torch.tensor([x], dtype=torch.float32)
        yt = torch.tensor([target], dtype=torch.float32)
        pred_t = torch_net(xt).squeeze(-1)
        loss_t = torch.nn.BCELoss()(pred_t, yt)
        loss_t.backward()

        # Compare output predictions
        assert math.isclose(pred_c.data, pred_t.item(), abs_tol=1e-5)
        # Compare loss values
        assert math.isclose(loss_c.data, loss_t.item(), abs_tol=1e-5)

        # Compare gradients of Layer 2
        l2_custom_w_grads = [w.grad for w in custom_net.layers[1].neurons[0].w]
        l2_torch_w_grads = torch_net.fc2.weight.grad[0].numpy().tolist()
        for cg, tg in zip(l2_custom_w_grads, l2_torch_w_grads):
            assert math.isclose(cg, tg, abs_tol=1e-4)

    def test_training_loss_reduction(self):
        """Verify that training on make_moons reduces loss over 20 epochs."""
        X_tr, X_te, y_tr, y_te, _ = prepare_moons_data(n_samples=100, random_state=42)
        res = train_custom_engine(X_tr, y_tr, X_te, y_te, epochs=20, lr=1.0, seed=42)

        history = res["history_loss"]
        # Initial loss should be higher than final loss
        assert history[-1] < history[0], (
            f"Loss failed to decrease: start={history[0]:.4f}, end={history[-1]:.4f}"
        )

    def test_reproducibility_with_fixed_seed(self):
        """Verify that training twice with the exact same seed yields identical results."""
        X_tr, X_te, y_tr, y_te, _ = prepare_moons_data(n_samples=50, random_state=42)

        run1 = train_custom_engine(X_tr, y_tr, X_te, y_te, epochs=10, lr=1.0, seed=123)
        run2 = train_custom_engine(X_tr, y_tr, X_te, y_te, epochs=10, lr=1.0, seed=123)

        assert math.isclose(run1["final_train_loss"], run2["final_train_loss"], abs_tol=1e-12)
        assert run1["final_train_acc"] == run2["final_train_acc"]
