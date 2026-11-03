"""
Core Automatic Differentiation Engine.

Implements the scalar `Value` node representing an operation in a dynamic
computation graph with reverse-mode automatic differentiation (backpropagation).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Set, Tuple, Union


Numeric = Union[int, float, "Value"]


class Value:
    """
    A scalar Value node representing an operand or intermediate result in a
    computation graph.

    Attributes:
        data (float): The scalar numerical value stored in this node.
        grad (float): The accumulated derivative d(Loss) / d(node).
        _prev (set[Value]): Set of parent/predecessor nodes that produced this node.
        _op (str): String identifier of the mathematical operation producing this node.
        label (str): Optional name or label for debugging and graph visualization.
        _backward (Callable[[], None]): Closure computing local derivatives and
            accumulating gradients back into predecessor nodes.
    """

    __slots__ = ("data", "grad", "_prev", "_op", "label", "_backward")

    def __init__(
        self,
        data: Union[int, float],
        _children: Tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        """
        Initialize a scalar Value node.

        Args:
            data: Numerical scalar value.
            _children: Predecessor Value nodes in the computation graph.
            _op: Operator symbol representing the generating operation.
            label: Human-readable label for debugging.
        """
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: Set[Value] = set(_children)
        self._op: str = _op
        self.label: str = label
        self._backward: Callable[[], None] = lambda: None

    # =========================================================================
    # Task 2: Arithmetic and Activation Operations
    # =========================================================================

    def __add__(self, other: Numeric) -> Value:
        """Addition: self + other."""
        other_node = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other_node.data, (self, other_node), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other_node.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Numeric) -> Value:
        """Right-addition: other + self."""
        return self.__add__(other)

    def __mul__(self, other: Numeric) -> Value:
        """Multiplication: self * other."""
        other_node = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other_node.data, (self, other_node), "*")

        def _backward() -> None:
            self.grad += other_node.data * out.grad
            other_node.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Numeric) -> Value:
        """Right-multiplication: other * self."""
        return self.__mul__(other)

    def __pow__(self, other: Union[int, float]) -> Value:
        """
        Power operation: self ** other, where exponent is a scalar int/float.
        Derivative: d(x^p)/dx = p * x^(p-1).
        """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Power exponent must be int or float, got {type(other).__name__}")

        p = float(other)
        out = Value(self.data ** p, (self,), f"**{p:g}")

        def _backward() -> None:
            # Handle power derivative gracefully
            if self.data == 0.0 and p < 1.0:
                # Avoid division by zero when data=0 and p-1 < 0
                deriv = 0.0
            else:
                deriv = p * (self.data ** (p - 1.0))
            self.grad += deriv * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """
        Rectified Linear Unit activation: max(0, x).
        Derivative: 1.0 if x > 0 else 0.0.
        """
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        """Negation: -self."""
        out = Value(-self.data, (self,), "-")

        def _backward() -> None:
            self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Numeric) -> Value:
        """Subtraction: self - other."""
        other_node = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other_node.data, (self, other_node), "-")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other_node.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: Numeric) -> Value:
        """Right-subtraction: other - self."""
        other_node = other if isinstance(other, Value) else Value(other)
        return other_node - self

    def __truediv__(self, other: Numeric) -> Value:
        """Division: self / other."""
        other_node = other if isinstance(other, Value) else Value(other)
        if other_node.data == 0.0:
            raise ZeroDivisionError("Division by zero in Value node evaluation.")
        out = Value(self.data / other_node.data, (self, other_node), "/")

        def _backward() -> None:
            self.grad += (1.0 / other_node.data) * out.grad
            other_node.grad += (-self.data / (other_node.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Numeric) -> Value:
        """Right-division: other / self."""
        other_node = other if isinstance(other, Value) else Value(other)
        return other_node / self

    def exp(self) -> Value:
        """
        Exponential: e^x.
        Derivative: d(e^x)/dx = e^x.
        """
        # Clamp large data to avoid float overflow
        clipped_data = min(self.data, 700.0)
        exp_val = math.exp(clipped_data)
        out = Value(exp_val, (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self, eps: float = 1e-15) -> Value:
        """
        Natural logarithm: ln(x).
        Uses numerical epsilon safeguarding for non-positive inputs.
        Derivative: d(ln(x))/dx = 1 / x.
        """
        val = max(self.data, eps)
        out = Value(math.log(val), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / val) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """
        Sigmoid activation function: 1 / (1 + exp(-x)).
        Derivative: sigmoid(x) * (1 - sigmoid(x)).
        """
        # Numerically stable sigmoid
        if self.data >= 0:
            s = 1.0 / (1.0 + math.exp(-self.data))
        else:
            exp_x = math.exp(self.data)
            s = exp_x / (1.0 + exp_x)

        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += (out.data * (1.0 - out.data)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """
        Hyperbolic tangent: tanh(x).
        Derivative: 1 - tanh(x)^2.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def sum(self) -> Value:
        """
        Sum operation on a scalar Value (returns itself as an identity node
        in the computation graph). Enables expression chaining:
        ((a * b + c) ** 2).relu().sum()
        """
        out = Value(self.data, (self,), "sum")

        def _backward() -> None:
            self.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    # =========================================================================
    # Task 3: Reverse-Mode Automatic Differentiation
    # =========================================================================

    def backward(self, grad: float = 1.0) -> None:
        """
        Execute reverse-mode automatic differentiation (backpropagation)
        from this node back to all ancestor nodes in the computation graph.

        Requirements:
        1. Topologically sorts the computation graph starting from self.
        2. Initializes the root gradient to 1.0 (or specified scalar grad).
        3. Traverses nodes in reverse topological order.
        4. Calculates local derivatives and accumulates gradients via +=
           to correctly handle shared parent nodes.

        Args:
            grad: Seed gradient at the output node (defaults to 1.0).
        """
        topo: list[Value] = []
        visited: Set[Value] = set()
        stack: list[Tuple[Value, bool]] = [(self, False)]

        # Iterative post-order DFS to construct topological sort safely
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for parent in node._prev:
                    if parent not in visited:
                        stack.append((parent, False))

        # Initialize output node gradient
        self.grad = float(grad)

        # Traverse nodes in reverse topological order
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Reset gradient of this node and all ancestor nodes to 0.0."""
        visited: Set[Value] = set()
        stack = [self]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                v.grad = 0.0
                stack.extend(v._prev)

    # =========================================================================
    # Representation and Utility
    # =========================================================================

    def __repr__(self) -> str:
        label_str = f", label='{self.label}'" if self.label else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f}{op_str}{label_str})"

    def __float__(self) -> float:
        return self.data

    def __int__(self) -> int:
        return int(self.data)
