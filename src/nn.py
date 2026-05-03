"""Tiny neural-network module library on top of the scalar autograd."""
from __future__ import annotations

import random

from src.autograd import Value


class Module:
    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, activation="relu"):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))
        self.activation = activation

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(
                f"Neuron expected input of length {len(self.w)}, got {len(x)}"
            )
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        if self.activation == "relu":
            return act.relu()
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "none" or self.activation is None:
            return act
        raise ValueError(f"unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, activation="relu"):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            is_last = i == len(nouts) - 1
            activation = "none" if is_last else "relu"
            layers.append(Layer(a, b, activation=activation))
        self.layers = layers

    def __call__(self, x):
        cur = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            cur = layer(cur)
        return cur

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
