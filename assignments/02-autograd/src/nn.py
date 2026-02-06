"""Neural network modules built on the scalar autograd engine."""

import random

from autograd import Value


class Neuron:
    """Single neuron: activation(sum(w_i * x_i) + b)."""

    def __init__(self, nin: int, activation: str = "relu"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return act.relu()
        elif self.activation == "tanh":
            return act.tanh()
        else:
            return act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """Layer of neurons."""

    def __init__(self, nin: int, nout: int, activation: str = "relu"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-layer perceptron. Last layer uses no activation (linear)."""

    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            act = "tanh" if i < len(nouts) - 1 else "linear"
            self.layers.append(Layer(sz[i], sz[i + 1], act))

    def __call__(self, x: list[float]) -> list[Value]:
        out = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
