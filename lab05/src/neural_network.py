"""
Neural Network Modules and Architectures.

Provides scalar-based neural network building blocks:
- Module: Base class with parameter management and gradient zeroing.
- Neuron: Single artificial neuron with configurable activation.
- Layer: Fully connected layer composed of neurons.
- MLP: Multi-layer perceptron (e.g. 2 -> 8 -> 1).
- PyTorchMLP: Exact equivalent PyTorch module for parity benchmarking and comparison.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from src.engine import Value
except ImportError:
    from .engine import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        """Zero out the accumulated gradients of all parameters."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        """Return a list of all trainable parameter Value nodes."""
        return []


class Neuron(Module):
    """
    A single artificial neuron computing y = act(sum(w_i * x_i) + b).
    """

    def __init__(
        self,
        nin: int,
        nonlin: str = "sigmoid",
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize neuron weights and bias.

        Args:
            nin: Number of incoming inputs.
            nonlin: Activation function ('sigmoid', 'relu', or 'linear').
            rng: NumPy random generator for deterministic initialization.
        """
        self.nin = nin
        self.nonlin = nonlin.lower()
        if rng is None:
            rng = np.random.default_rng()

        # Xavier / Glorot uniform initialization: [-bound, bound]
        # bound = sqrt(6 / (nin + 1))
        bound = math.sqrt(6.0 / (nin + 1.0))
        self.w: List[Value] = [
            Value(float(rng.uniform(-bound, bound)), label="w") for _ in range(nin)
        ]
        self.b: Value = Value(0.0, label="b")

    def __call__(self, x: Sequence[Union[float, Value]]) -> Value:
        """Compute the forward activation for input vector x."""
        if len(x) != self.nin:
            raise ValueError(
                f"Expected input dimension {self.nin}, got {len(x)}."
            )

        # sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.nonlin == "sigmoid":
            return act.sigmoid()
        elif self.nonlin == "relu":
            return act.relu()
        elif self.nonlin == "tanh":
            return act.tanh()
        elif self.nonlin in ("linear", "none", "identity"):
            return act
        else:
            raise ValueError(f"Unsupported activation: '{self.nonlin}'")

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron({self.nin}, nonlin='{self.nonlin}')"


class Layer(Module):
    """A fully connected dense layer composed of parallel neurons."""

    def __init__(
        self,
        nin: int,
        nout: int,
        nonlin: str = "sigmoid",
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize layer with nout neurons, each with nin inputs.
        """
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin, nonlin=nonlin, rng=rng) for _ in range(nout)]

    def __call__(self, x: Sequence[Union[float, Value]]) -> List[Value]:
        """Compute outputs for all neurons in the layer."""
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

    def __repr__(self) -> str:
        return f"Layer(nin={self.nin}, nout={self.nout}, neurons={len(self.neurons)})"


class MLP(Module):
    """
    Multi-Layer Perceptron (feed-forward neural network).
    Default architecture: 2 -> 8 -> 1 with Sigmoid activations.
    """

    def __init__(
        self,
        nin: int = 2,
        nouts: Sequence[int] = (8, 1),
        activations: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize multi-layer perceptron.

        Args:
            nin: Number of input features.
            nouts: List/tuple of layer output dimensions (e.g. [8, 1]).
            activations: List of activation functions for each layer.
                         Defaults to ['sigmoid', 'sigmoid'].
            seed: Random seed for parameter initialization.
        """
        rng = np.random.default_rng(seed)
        sz = [nin] + list(nouts)

        if activations is None:
            # Default to sigmoid for all layers (as specified in Task 5)
            activations = ["sigmoid"] * len(nouts)
        elif len(activations) != len(nouts):
            raise ValueError("Number of activations must match number of layers.")

        self.layers: List[Layer] = []
        for i in range(len(nouts)):
            self.layers.append(
                Layer(sz[i], sz[i + 1], nonlin=activations[i], rng=rng)
            )

    def __call__(self, x: Sequence[Union[float, Value]]) -> Union[Value, List[Value]]:
        """
        Forward pass for a single input vector.
        Returns a single Value if output layer has size 1, else a list of Values.
        """
        out: Sequence[Union[float, Value]] = x
        for layer in self.layers:
            out = layer(out)
        return out[0] if len(out) == 1 else list(out)

    def forward_batch(self, X: Sequence[Sequence[float]]) -> List[Value]:
        """
        Forward pass for a collection of input vectors.
        Returns a list of scalar predictions (Value nodes).
        """
        return [self(xi) for xi in X]  # type: ignore

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def get_weights_numpy(self) -> List[np.ndarray]:
        """Extract parameter values as NumPy arrays per layer."""
        arrays = []
        for layer in self.layers:
            w_mat = np.array([[w.data for w in n.w] for n in layer.neurons], dtype=np.float64)
            b_vec = np.array([n.b.data for n in layer.neurons], dtype=np.float64)
            arrays.append((w_mat, b_vec))
        return arrays

    def set_weights_numpy(self, layer_weights: List[tuple[np.ndarray, np.ndarray]]) -> None:
        """Set parameters from NumPy arrays."""
        for layer, (w_mat, b_vec) in zip(self.layers, layer_weights):
            for i, neuron in enumerate(layer.neurons):
                for j, w_node in enumerate(neuron.w):
                    w_node.data = float(w_mat[i, j])
                neuron.b.data = float(b_vec[i])

    def __repr__(self) -> str:
        return f"MLP(layers={self.layers})"


class PyTorchMLP(nn.Module):
    """
    Equivalent PyTorch Neural Network module with 2 -> 8 -> 1 architecture.
    """

    def __init__(
        self,
        nin: int = 2,
        hidden_dim: int = 8,
        nout: int = 1,
        hidden_activation: str = "sigmoid",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.nin = nin
        self.hidden_dim = hidden_dim
        self.nout = nout
        self.hidden_activation_name = hidden_activation.lower()

        self.fc1 = nn.Linear(nin, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, nout, bias=True)

        # Xavier uniform initialization matching custom engine
        bound1 = math.sqrt(6.0 / (nin + 1.0))
        nn.init.uniform_(self.fc1.weight, -bound1, bound1)
        nn.init.zeros_(self.fc1.bias)

        bound2 = math.sqrt(6.0 / (hidden_dim + 1.0))
        nn.init.uniform_(self.fc2.weight, -bound2, bound2)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x -> fc1 -> sigmoid -> fc2 -> sigmoid -> out
        """
        if self.hidden_activation_name == "sigmoid":
            h = torch.sigmoid(self.fc1(x))
        elif self.hidden_activation_name == "relu":
            h = torch.relu(self.fc1(x))
        else:
            h = self.fc1(x)

        out = torch.sigmoid(self.fc2(h))
        return out

    def sync_weights_from_custom(self, custom_mlp: MLP) -> None:
        """
        Copy weights from custom MLP into this PyTorch module for exact parity tests.
        """
        with torch.no_grad():
            # Layer 1
            l1_neurons = custom_mlp.layers[0].neurons
            w1 = torch.tensor([[w.data for w in n.w] for n in l1_neurons], dtype=torch.float32)
            b1 = torch.tensor([n.b.data for n in l1_neurons], dtype=torch.float32)
            self.fc1.weight.copy_(w1)
            self.fc1.bias.copy_(b1)

            # Layer 2
            l2_neurons = custom_mlp.layers[1].neurons
            w2 = torch.tensor([[w.data for w in n.w] for n in l2_neurons], dtype=torch.float32)
            b2 = torch.tensor([n.b.data for n in l2_neurons], dtype=torch.float32)
            self.fc2.weight.copy_(w2)
            self.fc2.bias.copy_(b2)

    def sync_weights_to_custom(self, custom_mlp: MLP) -> None:
        """
        Copy weights from this PyTorch module to custom MLP.
        """
        with torch.no_grad():
            # Layer 1
            w1 = self.fc1.weight.cpu().numpy()
            b1 = self.fc1.bias.cpu().numpy()
            for i, neuron in enumerate(custom_mlp.layers[0].neurons):
                for j, w in enumerate(neuron.w):
                    w.data = float(w1[i, j])
                neuron.b.data = float(b1[i])

            # Layer 2
            w2 = self.fc2.weight.cpu().numpy()
            b2 = self.fc2.bias.cpu().numpy()
            for i, neuron in enumerate(custom_mlp.layers[1].neurons):
                for j, w in enumerate(neuron.w):
                    w.data = float(w2[i, j])
                neuron.b.data = float(b2[i])
