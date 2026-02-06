"""Scalar autograd engine with reverse-mode autodiff."""

from __future__ import annotations

import math


class Value:
    """Wraps a scalar float and tracks computation graph for autodiff."""

    def __init__(self, data: float, label: str = "") -> None:
        self.data = float(data)
        self.label = label
        self.grad = 0.0
        self._backward: callable = lambda: None
        self._prev: set[Value] = set()

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: int | float) -> Value:
        return self * other

    def __pow__(self, other: int | float) -> Value:
        out = Value(self.data ** other)
        out._prev = {self}

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __truediv__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def relu(self) -> Value:
        out = Value(max(0.0, self.data))
        out._prev = {self}

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t)
        out._prev = {self}

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e)
        out._prev = {self}

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad += 1.0
        for v in reversed(topo):
            v._backward()
