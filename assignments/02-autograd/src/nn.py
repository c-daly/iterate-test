"""Tiny neural-network library built on the scalar autograd Value class.

Provides Neuron, Layer, and MLP. The MLP follows the convention that the
final layer applies no activation (raw linear output suitable for both
regression and as logits before an external loss).
"""
from __future__ import annotations

import random
from typing import List

from .autograd import Value

_VALID_ACTIVATIONS = ("relu", "tanh", "none")


class Neuron:
    """Single neuron: dot(weights, inputs) + bias, then optional activation."""

    def __init__(self, nin: int, activation: str = "relu") -> None:
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_VALID_ACTIVATIONS}, got {activation!r}"
            )
        self.activation: str = activation
        # Uniform init in [-1, 1] is standard for tiny scalar nets.
        self.w: List[Value] = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b: Value = Value(0.0)

    def __call__(self, x: List[Value]) -> Value:
        # Pre-activation: sum(w_i * x_i) + b. Use Value(0.0) as the start to
        # keep the result inside the autograd graph regardless of nin.
        act: Value = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        if self.activation == "relu":
            return act.relu()
        if self.activation == "tanh":
            return act.tanh()
        return act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    """A layer is a collection of independent neurons sharing input width."""

    def __init__(self, nin: int, nout: int, activation: str = "relu") -> None:
        self.neurons: List[Neuron] = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class MLP:
    """Multi-layer perceptron. The final layer uses no activation."""

    def __init__(self, nin: int, nouts: List[int]) -> None:
        if not nouts:
            raise ValueError("nouts must contain at least one layer width")
        sizes = [nin] + list(nouts)
        self.layers: List[Layer] = []
        last_index = len(nouts) - 1
        for i, nout in enumerate(nouts):
            activation = "none" if i == last_index else "relu"
            self.layers.append(Layer(sizes[i], nout, activation=activation))

    def __call__(self, x: List[float]) -> List[Value]:
        # Convert all raw floats to Values at the boundary; if the caller
        # already passed Values they are forwarded unchanged.
        current: List[Value] = [xi if isinstance(xi, Value) else Value(float(xi)) for xi in x]
        for layer in self.layers:
            current = layer(current)
        return current

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
