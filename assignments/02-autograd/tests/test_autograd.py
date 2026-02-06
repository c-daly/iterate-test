"""Comprehensive tests for scalar autograd engine and neural network modules."""

import math
import random


import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autograd import Value
from nn import Neuron, Layer, MLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_grad(f, x, eps=1e-6):
    """Compute numerical gradient of f at x using central differences."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def check_grad(value_expr_fn, wrt_data, eps=1e-6, tol=1e-4):
    """Build expression, backprop, compare analytic vs numerical gradient.

    value_expr_fn: callable(Value) -> Value  (builds expression from a single input)
    wrt_data: float value to test at
    """
    # Analytic
    v = Value(wrt_data)
    out = value_expr_fn(v)
    out.backward()
    analytic = v.grad

    # Numerical
    def f(x):
        return value_expr_fn(Value(x)).data

    num = numerical_grad(f, wrt_data, eps)
    assert abs(analytic - num) < tol, (
        f"Gradient mismatch: analytic={analytic}, numerical={num}"
    )


# ===========================================================================
# Value construction
# ===========================================================================

class TestValueBasics:
    def test_create_value(self):
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0

    def test_create_value_with_label(self):
        v = Value(3.0, label="x")
        assert v.label == "x"

    def test_repr(self):
        v = Value(2.5)
        r = repr(v)
        assert "2.5" in r


# ===========================================================================
# Individual operations — forward + gradient
# ===========================================================================

class TestAdd:
    def test_forward(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_add_scalar_right(self):
        a = Value(2.0)
        c = a + 5
        assert c.data == 7.0
        c.backward()
        assert a.grad == 1.0

    def test_radd(self):
        a = Value(2.0)
        c = 5 + a
        assert c.data == 7.0
        c.backward()
        assert a.grad == 1.0

    def test_add_float(self):
        a = Value(2.0)
        c = a + 3.5
        assert c.data == 5.5


class TestMul:
    def test_forward(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_mul_scalar_right(self):
        a = Value(4.0)
        c = a * 3
        assert c.data == 12.0
        c.backward()
        assert a.grad == 3.0

    def test_rmul(self):
        a = Value(4.0)
        c = 3 * a
        assert c.data == 12.0
        c.backward()
        assert a.grad == 3.0


class TestPow:
    def test_forward(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_gradient_square(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert abs(a.grad - 6.0) < 1e-6

    def test_gradient_cube(self):
        a = Value(2.0)
        c = a ** 3
        c.backward()
        assert abs(a.grad - 12.0) < 1e-6  # 3 * 2^2 = 12

    def test_pow_float(self):
        a = Value(4.0)
        c = a ** 0.5
        assert abs(c.data - 2.0) < 1e-6

    def test_gradient_numerical(self):
        check_grad(lambda v: v ** 3, 2.5)


class TestNeg:
    def test_forward(self):
        a = Value(3.0)
        c = -a
        assert c.data == -3.0

    def test_gradient(self):
        a = Value(3.0)
        c = -a
        c.backward()
        assert a.grad == -1.0


class TestSub:
    def test_forward(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_gradient(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_sub_scalar(self):
        a = Value(5.0)
        c = a - 2
        assert c.data == 3.0

    def test_rsub(self):
        a = Value(3.0)
        c = 10 - a
        assert c.data == 7.0
        c.backward()
        assert a.grad == -1.0


class TestDiv:
    def test_forward(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-6

    def test_gradient(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        c.backward()
        assert abs(a.grad - 1.0 / 3.0) < 1e-6
        assert abs(b.grad - (-6.0 / 9.0)) < 1e-6

    def test_div_scalar(self):
        a = Value(10.0)
        c = a / 2
        assert abs(c.data - 5.0) < 1e-6

    def test_rdiv(self):
        a = Value(4.0)
        c = 8 / a
        assert abs(c.data - 2.0) < 1e-6
        c.backward()
        # d/da (8/a) = -8/a^2 = -0.5
        assert abs(a.grad - (-0.5)) < 1e-6


# ===========================================================================
# Activations
# ===========================================================================

class TestRelu:
    def test_positive(self):
        a = Value(3.0)
        c = a.relu()
        assert c.data == 3.0
        c.backward()
        assert a.grad == 1.0

    def test_negative(self):
        a = Value(-3.0)
        c = a.relu()
        assert c.data == 0.0
        c.backward()
        assert a.grad == 0.0

    def test_zero(self):
        a = Value(0.0)
        c = a.relu()
        assert c.data == 0.0

    def test_numerical_grad(self):
        check_grad(lambda v: v.relu(), 2.0)


class TestTanh:
    def test_forward(self):
        a = Value(0.0)
        c = a.tanh()
        assert abs(c.data - 0.0) < 1e-6

    def test_forward_positive(self):
        a = Value(1.0)
        c = a.tanh()
        assert abs(c.data - math.tanh(1.0)) < 1e-6

    def test_gradient(self):
        a = Value(0.5)
        c = a.tanh()
        c.backward()
        expected = 1.0 - math.tanh(0.5) ** 2
        assert abs(a.grad - expected) < 1e-6

    def test_numerical_grad(self):
        check_grad(lambda v: v.tanh(), 0.8)


class TestExp:
    def test_forward(self):
        a = Value(1.0)
        c = a.exp()
        assert abs(c.data - math.e) < 1e-6

    def test_gradient(self):
        a = Value(2.0)
        c = a.exp()
        c.backward()
        assert abs(a.grad - math.exp(2.0)) < 1e-6

    def test_numerical_grad(self):
        check_grad(lambda v: v.exp(), 1.5)


# ===========================================================================
# Chain rule — multi-step expressions
# ===========================================================================

class TestChainRule:
    def test_simple_chain(self):
        """f(x) = (x * 2 + 1)^2, df/dx at x=3 => 2*(3*2+1)*2 = 28"""
        x = Value(3.0)
        f = (x * 2 + 1) ** 2
        f.backward()
        assert abs(x.grad - 28.0) < 1e-6

    def test_longer_chain(self):
        """f(x) = tanh(x^2 + 1), numerical check"""
        check_grad(lambda v: (v ** 2 + 1).tanh(), 0.5)

    def test_multi_input(self):
        """f(x,y) = x*y + x^2, df/dx = y+2x, df/dy = x"""
        x = Value(3.0)
        y = Value(4.0)
        f = x * y + x ** 2
        f.backward()
        assert abs(x.grad - (4.0 + 6.0)) < 1e-6  # y + 2x
        assert abs(y.grad - 3.0) < 1e-6  # x

    def test_complex_expression(self):
        """Test a more complex expression numerically."""
        check_grad(lambda v: ((v * 2).relu() + (v ** 2)).tanh(), 1.0)


# ===========================================================================
# Gradient accumulation (diamond graphs)
# ===========================================================================

class TestGradientAccumulation:
    def test_value_used_twice_add(self):
        """f(x) = x + x = 2x, df/dx = 2"""
        x = Value(3.0)
        f = x + x
        f.backward()
        assert abs(x.grad - 2.0) < 1e-6

    def test_value_used_twice_mul(self):
        """f(x) = x * x = x^2, df/dx = 2x"""
        x = Value(3.0)
        f = x * x
        f.backward()
        assert abs(x.grad - 6.0) < 1e-6

    def test_diamond_graph(self):
        """x -> a,b -> c where a=x+1, b=x*2, c=a*b
        dc/dx = dc/da * da/dx + dc/db * db/dx = b*1 + a*2
        At x=3: a=4, b=6, dc/dx = 6 + 8 = 14
        """
        x = Value(3.0)
        a = x + 1
        b = x * 2
        c = a * b
        c.backward()
        assert abs(x.grad - 14.0) < 1e-6

    def test_triple_use(self):
        """f(x) = x + x + x = 3x, df/dx = 3"""
        x = Value(5.0)
        f = x + x + x
        f.backward()
        assert abs(x.grad - 3.0) < 1e-6


# ===========================================================================
# Numerical gradient verification for composed expressions
# ===========================================================================

class TestNumericalGradients:
    def test_polynomial(self):
        check_grad(lambda v: v ** 3 + v ** 2 + v + Value(1.0), 2.0)

    def test_division_chain(self):
        check_grad(lambda v: Value(1.0) / (v + Value(1.0)), 2.0)

    def test_exp_composition(self):
        check_grad(lambda v: (v * Value(-1.0)).exp(), 1.0)

    def test_tanh_of_product(self):
        check_grad(lambda v: (v * Value(2.0)).tanh(), 0.5)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_zero_gradient_relu(self):
        """ReLU with negative input has zero gradient."""
        x = Value(-5.0)
        f = x.relu()
        f.backward()
        assert x.grad == 0.0

    def test_backward_sets_root_grad_to_one(self):
        x = Value(3.0)
        y = x * 2
        y.backward()
        assert y.grad == 1.0

    def test_repeated_backward_accumulates(self):
        """Calling backward twice should accumulate gradients."""
        x = Value(3.0)
        y = x * 2
        y.backward()
        assert x.grad == 2.0
        y.backward()
        assert x.grad == 4.0  # accumulated

    def test_no_grad_before_backward(self):
        x = Value(3.0)
        _ = x * 2
        assert x.grad == 0.0

    def test_pow_zero(self):
        a = Value(5.0)
        c = a ** 0
        assert c.data == 1.0

    def test_add_int_left(self):
        """int + Value should work via __radd__."""
        v = Value(2.0)
        c = 3 + v
        assert c.data == 5.0


# ===========================================================================
# Neural network modules
# ===========================================================================

class TestNeuron:
    def test_creation(self):
        n = Neuron(3)
        params = n.parameters()
        assert len(params) == 4  # 3 weights + 1 bias

    def test_forward(self):
        random.seed(42)
        n = Neuron(2, activation="relu")
        x = [Value(1.0), Value(2.0)]
        out = n(x)
        assert isinstance(out, Value)

    def test_tanh_activation(self):
        random.seed(42)
        n = Neuron(2, activation="tanh")
        x = [Value(1.0), Value(2.0)]
        out = n(x)
        assert isinstance(out, Value)
        assert -1.0 <= out.data <= 1.0

    def test_no_activation(self):
        random.seed(42)
        n = Neuron(2, activation="linear")
        x = [Value(1.0), Value(2.0)]
        out = n(x)
        assert isinstance(out, Value)
        # linear output can be anything — just check it's a Value


class TestLayer:
    def test_creation(self):
        layer = Layer(3, 4)
        params = layer.parameters()
        assert len(params) == 4 * (3 + 1)  # 4 neurons, each with 3 weights + 1 bias

    def test_forward(self):
        random.seed(42)
        layer = Layer(2, 3)
        x = [Value(1.0), Value(2.0)]
        out = layer(x)
        assert len(out) == 3
        assert all(isinstance(v, Value) for v in out)


class TestMLP:
    def test_creation(self):
        mlp = MLP(3, [4, 2])
        params = mlp.parameters()
        # Layer 1: 4 neurons * (3+1) = 16
        # Layer 2: 2 neurons * (4+1) = 10
        assert len(params) == 26

    def test_forward(self):
        random.seed(42)
        mlp = MLP(2, [3, 1])
        out = mlp([1.0, 2.0])
        assert len(out) == 1
        assert isinstance(out[0], Value)

    def test_gradients_flow(self):
        """Ensure backward pass through MLP updates all parameter gradients."""
        random.seed(42)
        mlp = MLP(2, [3, 1])
        out = mlp([1.0, 2.0])
        loss = out[0] ** 2
        loss.backward()
        # At least some parameters should have non-zero gradients
        grads = [p.grad for p in mlp.parameters()]
        assert any(g != 0.0 for g in grads)


# ===========================================================================
# XOR learning convergence
# ===========================================================================

class TestXORLearning:
    def test_xor_convergence(self):
        """Train a small MLP on XOR, verify loss decreases."""
        random.seed(42)
        mlp = MLP(2, [4, 4, 1])

        xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        ys = [0.0, 1.0, 1.0, 0.0]

        initial_loss = None
        lr = 0.05

        for epoch in range(200):
            # Forward
            preds = [mlp(x)[0] for x in xs]
            loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0))

            if epoch == 0:
                initial_loss = loss.data

            # Backward
            for p in mlp.parameters():
                p.grad = 0.0
            loss.backward()

            # Update
            for p in mlp.parameters():
                p.data -= lr * p.grad

        final_loss = loss.data
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
        )
        # Loss should be reasonably small after 200 epochs
        assert final_loss < 0.5, f"Final loss too high: {final_loss}"
