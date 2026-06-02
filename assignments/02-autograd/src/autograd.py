"""Scalar-valued automatic differentiation engine (reverse-mode autodiff).

The `Value` class wraps a single Python float and records the computation
graph as operations are applied. Each operation produces a new `Value` whose
``_backward`` closure pushes the upstream gradient onto its inputs (the local
derivative times the incoming gradient). Calling ``backward()`` on an output
node topologically sorts the graph and invokes those closures in reverse
order, accumulating gradients into each node's ``grad`` attribute.

Pure Python only -- no numpy / torch.
"""

import math


class Value:
    """A scalar value node in an autodiff computation graph."""

    def __init__(self, data, label="", _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def _coerce(self, other):
        """Wrap a raw int/float operand in a Value; pass Values through."""
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._coerce(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._coerce(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "__pow__ supports int/float exponents only"
        out = Value(self.data ** other, _children=(self,), _op=f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __truediv__(self, other):
        return self * self._coerce(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, _children=(self,), _op="relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, _children=(self,), _op="exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Topologically sort the graph and back-propagate gradients.

        Seeds the output node with ``grad += 1`` (dout/dout = 1) and invokes
        each node's ``_backward`` closure in reverse topological order,
        accumulating into every input's ``grad``. Gradients are intentionally
        not zeroed here, so callers manage zeroing between independent passes.
        """
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        label = f" label={self.label!r}" if self.label else ""
        return f"Value(data={self.data}, grad={self.grad}{label})"
