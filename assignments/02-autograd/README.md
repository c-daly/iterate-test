# Assignment 2: Scalar Autograd Engine

## Overview

Implement a scalar-valued automatic differentiation engine using reverse-mode autodiff (backpropagation). Build a small neural network library on top of it.

## Requirements

### Core: `Value` class

File: `src/autograd.py`

```python
class Value:
    def __init__(self, data: float, label: str = ""): ...
    def __add__(self, other): ...
    def __mul__(self, other): ...
    def __pow__(self, other: int | float): ...
    def __neg__(self): ...
    def __sub__(self, other): ...
    def __truediv__(self, other): ...
    def __radd__(self, other): ...
    def __rmul__(self, other): ...
    def relu(self) -> "Value": ...
    def tanh(self) -> "Value": ...
    def exp(self) -> "Value": ...
    def backward(self) -> None: ...
```

### Neural Network Modules

File: `src/nn.py`

```python
class Neuron:
    def __init__(self, nin: int, activation: str = "relu"): ...
    def __call__(self, x: list[Value]) -> Value: ...
    def parameters(self) -> list[Value]: ...

class Layer:
    def __init__(self, nin: int, nout: int, activation: str = "relu"): ...
    def __call__(self, x: list[Value]) -> list[Value]: ...
    def parameters(self) -> list[Value]: ...

class MLP:
    def __init__(self, nin: int, nouts: list[int]): ...
    def __call__(self, x: list[float]) -> list[Value]: ...
    def parameters(self) -> list[Value]: ...
```

### Behavior

- `Value` wraps a scalar float and tracks computation graph.
- All arithmetic ops return new `Value` nodes with `_backward` closures.
- `backward()` performs reverse-mode autodiff via topological sort.
- Gradients accumulate (important for shared nodes).
- `Neuron` computes `activation(sum(w*x) + b)`.
- `MLP` chains layers; last layer uses no activation.

### Constraints

- No numpy/torch — pure Python arithmetic only.
- `backward()` must handle diamond-shaped graphs (gradient accumulation).
- Support `int` and `float` on left side of operators (`2 * value`).

## Test Expectations

Tests in `tests/test_autograd.py` should cover:
- Individual ops: add, mul, pow, neg, sub, div with gradient checks
- Activations: relu, tanh, exp
- Chain rule through multi-step expressions
- Gradient accumulation (same value used twice)
- Numerical gradient verification (finite differences)
- Neuron/Layer/MLP forward pass
- XOR learning convergence (train small MLP, verify loss decreases)
- Edge cases: zero gradients, repeated backward calls
