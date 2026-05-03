"""Scalar-valued reverse-mode automatic differentiation.

This module implements a small autograd engine. Each Value wraps a Python
float and records the operations that produced it so gradients can be
propagated backwards via Value.backward. Gradient accumulation (using +=)
ensures correctness for diamond-shaped computation graphs where the same
node feeds multiple children.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple, Union

Number = Union[int, float]


class Value:
    """A scalar node in a computation graph with a retained backward closure."""

    __slots__ = ("data", "grad", "label", "_prev", "_op", "_backward")

    def __init__(
        self,
        data: float,
        label: str = "",
        _children: Iterable["Value"] = (),
        _op: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self.label: str = label
        self._prev: Tuple[Value, ...] = tuple(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = _noop_backward

    def __repr__(self) -> str:
        label_part = f" label={self.label!r}" if self.label else ""
        op_part = f" op={self._op!r}" if self._op else ""
        return f"Value(data={self.data}, grad={self.grad}{label_part}{op_part})"

    @staticmethod
    def _as_value(other: Union["Value", Number]) -> "Value":
        if isinstance(other, Value):
            return other
        return Value(float(other))

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = self._as_value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Number) -> "Value":
        return self.__add__(other)

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        other = self._as_value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Number) -> "Value":
        return self.__mul__(other)

    def __pow__(self, exponent: Number) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("exponent must be int or float, got " + type(exponent).__name__)
        out = Value(self.data ** exponent, _children=(self,), _op=f"**{exponent}")

        def _backward() -> None:
            # d/dx x**n = n * x**(n-1). When n == 0 the derivative is 0
            # everywhere it is defined; skip the n*x**(n-1) computation to
            # avoid 0 * 0**-1 -> ZeroDivisionError at x == 0.
            if exponent != 0:
                self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        return self + (-self._as_value(other))

    def __rsub__(self, other: Number) -> "Value":
        return self._as_value(other) - self

    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        return self * (self._as_value(other) ** -1)

    def __rtruediv__(self, other: Number) -> "Value":
        return self._as_value(other) * (self ** -1)

    def relu(self) -> "Value":
        out_data = self.data if self.data > 0.0 else 0.0
        out = Value(out_data, _children=(self,), _op="relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, _children=(self,), _op="tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        e = math.exp(self.data)
        out = Value(e, _children=(self,), _op="exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Run reverse-mode autodiff from this node, seeding grad = 1.0.

        Gradients accumulate into Value.grad; callers that re-use a graph
        across multiple loss evaluations are responsible for resetting grad
        to 0.0 between calls (mirrors the zero_grad pattern from PyTorch).
        """
        topo: list[Value] = []
        visited: set[int] = set()

        def build(v: Value) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad += 1.0
        for node in reversed(topo):
            node._backward()


def _noop_backward() -> None:
    return None
