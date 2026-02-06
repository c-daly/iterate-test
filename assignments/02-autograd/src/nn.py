"""Neural network modules built on the autograd engine."""

from __future__ import annotations

import random

from autograd import Value


class Neuron:
    """Single neuron with weights, bias, and activation."""

    def __init__(self, nin: int, activation: str = "relu") -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return act.relu()
        elif self.activation == "tanh":
            return act.tanh()
        elif self.activation == "linear":
            return act
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """Layer of neurons."""

    def __init__(
        self, nin: int, nout: int, activation: str = "relu"
    ) -> None:
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-layer perceptron. Last layer uses no activation."""

    def __init__(self, nin: int, nouts: list[int]) -> None:
        sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            act = "relu" if i < len(nouts) - 1 else "linear"
            self.layers.append(Layer(sizes[i], sizes[i + 1], activation=act))

    def __call__(self, x: list[float]) -> list[Value]:
        out: list[Value] = [Value(xi) for xi in x]
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
