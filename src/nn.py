"""Neural network modules built on the scalar autograd engine."""

import random
from src.autograd import Value


class Neuron:
    """Single neuron: computes activation(sum(w*x) + b)."""

    def __init__(self, nin: int, activation: str = "relu"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: list) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return act.relu()
        elif self.activation == "tanh":
            return act.tanh()
        else:
            return act

    def parameters(self) -> list:
        return self.w + [self.b]


class Layer:
    """Layer of neurons."""

    def __init__(self, nin: int, nout: int, activation: str = "relu"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x: list) -> list:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-layer perceptron. Last layer uses no activation."""

    def __init__(self, nin: int, nouts: list):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            activation = "relu" if i < len(nouts) - 1 else ""
            self.layers.append(Layer(sz[i], sz[i + 1], activation))

    def __call__(self, x: list) -> list:
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list:
        return [p for layer in self.layers for p in layer.parameters()]
