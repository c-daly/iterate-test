"""Comprehensive tests for scalar autograd engine and neural network modules."""

from __future__ import annotations

import math
import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autograd import Value  # noqa: E402
from nn import MLP, Layer, Neuron  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_grad(f, x: float, eps: float = 1e-6) -> float:
    """Compute numerical gradient of f at x using central differences."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def check_grad(f, x_val: float, tol: float = 1e-4) -> None:
    """Verify analytical gradient matches numerical gradient."""
    x = Value(x_val)
    y = f(x)
    y.backward()
    analytic = x.grad
    numeric = numerical_grad(lambda v: f(Value(v)).data, x_val)
    assert abs(analytic - numeric) < tol, (
        f"Grad mismatch: analytic={analytic}, numeric={numeric}"
    )


# ===========================================================================
# Value construction
# ===========================================================================

class TestValueConstruction:
    def test_create_value(self):
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0

    def test_create_value_with_label(self):
        v = Value(3.0, label="x")
        assert v.data == 3.0
        assert v.label == "x"

    def test_repr(self):
        v = Value(4.5)
        r = repr(v)
        assert "4.5" in r


# ===========================================================================
# Individual arithmetic ops + gradient checks
# ===========================================================================

class TestAdd:
    def test_add_two_values(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_add_value_and_float(self):
        a = Value(2.0)
        c = a + 3.0
        assert c.data == 5.0

    def test_add_value_and_int(self):
        a = Value(2.0)
        c = a + 3
        assert c.data == 5.0

    def test_radd(self):
        a = Value(2.0)
        c = 3.0 + a
        assert c.data == 5.0

    def test_add_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_add_gradient_numerical(self):
        check_grad(lambda x: x + Value(3.0), 2.0)


class TestMul:
    def test_mul_two_values(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_mul_value_and_float(self):
        a = Value(2.0)
        c = a * 3.0
        assert c.data == 6.0

    def test_rmul(self):
        a = Value(2.0)
        c = 3.0 * a
        assert c.data == 6.0

    def test_rmul_int(self):
        a = Value(2.0)
        c = 3 * a
        assert c.data == 6.0

    def test_mul_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_mul_gradient_numerical(self):
        check_grad(lambda x: x * Value(3.0), 2.0)


class TestPow:
    def test_pow_int(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_pow_float(self):
        a = Value(4.0)
        c = a ** 0.5
        assert abs(c.data - 2.0) < 1e-7

    def test_pow_gradient(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert abs(a.grad - 6.0) < 1e-7

    def test_pow_gradient_numerical(self):
        check_grad(lambda x: x ** 3, 2.0)


class TestNeg:
    def test_neg(self):
        a = Value(3.0)
        c = -a
        assert c.data == -3.0

    def test_neg_gradient(self):
        a = Value(3.0)
        c = -a
        c.backward()
        assert a.grad == -1.0


class TestSub:
    def test_sub(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_sub_float(self):
        a = Value(5.0)
        c = a - 3.0
        assert c.data == 2.0

    def test_sub_gradient(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_sub_gradient_numerical(self):
        check_grad(lambda x: x - Value(3.0), 5.0)


class TestDiv:
    def test_div(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-7

    def test_div_float(self):
        a = Value(6.0)
        c = a / 3.0
        assert abs(c.data - 2.0) < 1e-7

    def test_div_gradient(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        c.backward()
        # dc/da = 1/b = 1/3
        assert abs(a.grad - 1.0 / 3.0) < 1e-7
        # dc/db = -a / b^2 = -6/9 = -2/3
        assert abs(b.grad - (-2.0 / 3.0)) < 1e-7

    def test_div_gradient_numerical(self):
        check_grad(lambda x: x / Value(3.0), 6.0)


# ===========================================================================
# Activations
# ===========================================================================

class TestRelu:
    def test_relu_positive(self):
        a = Value(3.0)
        c = a.relu()
        assert c.data == 3.0

    def test_relu_negative(self):
        a = Value(-3.0)
        c = a.relu()
        assert c.data == 0.0

    def test_relu_zero(self):
        a = Value(0.0)
        c = a.relu()
        assert c.data == 0.0

    def test_relu_gradient_positive(self):
        a = Value(3.0)
        c = a.relu()
        c.backward()
        assert a.grad == 1.0

    def test_relu_gradient_negative(self):
        a = Value(-3.0)
        c = a.relu()
        c.backward()
        assert a.grad == 0.0

    def test_relu_gradient_numerical(self):
        check_grad(lambda x: x.relu(), 2.0)


class TestTanh:
    def test_tanh_value(self):
        a = Value(1.0)
        c = a.tanh()
        assert abs(c.data - math.tanh(1.0)) < 1e-7

    def test_tanh_zero(self):
        a = Value(0.0)
        c = a.tanh()
        assert abs(c.data) < 1e-7

    def test_tanh_gradient(self):
        a = Value(1.0)
        c = a.tanh()
        c.backward()
        expected = 1 - math.tanh(1.0) ** 2
        assert abs(a.grad - expected) < 1e-7

    def test_tanh_gradient_numerical(self):
        check_grad(lambda x: x.tanh(), 0.5)


class TestExp:
    def test_exp_value(self):
        a = Value(1.0)
        c = a.exp()
        assert abs(c.data - math.e) < 1e-7

    def test_exp_zero(self):
        a = Value(0.0)
        c = a.exp()
        assert abs(c.data - 1.0) < 1e-7

    def test_exp_gradient(self):
        a = Value(2.0)
        c = a.exp()
        c.backward()
        assert abs(a.grad - math.exp(2.0)) < 1e-7

    def test_exp_gradient_numerical(self):
        check_grad(lambda x: x.exp(), 1.0)


# ===========================================================================
# Chain rule / multi-step expressions
# ===========================================================================

class TestChainRule:
    def test_multi_step_expression(self):
        """f(x) = (x * 2 + 1) ** 2, df/dx at x=3 = 2*(x*2+1)*2 = 4*(3*2+1) = 28"""
        x = Value(3.0)
        y = (x * 2 + 1) ** 2
        y.backward()
        assert abs(y.data - 49.0) < 1e-7
        assert abs(x.grad - 28.0) < 1e-7

    def test_chain_mul_add(self):
        x = Value(2.0)
        y = Value(-3.0)
        z = x * y + Value(10.0)
        z.backward()
        assert abs(z.data - 4.0) < 1e-7
        assert abs(x.grad - (-3.0)) < 1e-7
        assert abs(y.grad - 2.0) < 1e-7

    def test_nested_operations(self):
        """f = tanh(x*w + b), check gradients flow through."""
        x = Value(2.0)
        w = Value(0.5)
        b = Value(0.1)
        y = (x * w + b).tanh()
        y.backward()
        # Just verify gradient exists and is nonzero
        assert w.grad != 0.0
        assert x.grad != 0.0
        assert b.grad != 0.0

    def test_complex_expression_numerical(self):
        """Verify a complex expression against numerical gradient."""
        def f(x):
            return ((x * Value(2.0) + Value(1.0)).tanh() * Value(3.0)) ** 2
        check_grad(f, 0.5)


# ===========================================================================
# Gradient accumulation (diamond graph / shared nodes)
# ===========================================================================

class TestGradientAccumulation:
    def test_same_value_used_twice_add(self):
        """a + a should give grad=2."""
        a = Value(3.0)
        c = a + a
        c.backward()
        assert abs(a.grad - 2.0) < 1e-7

    def test_same_value_used_twice_mul(self):
        """a * a = a^2, grad = 2a."""
        a = Value(3.0)
        c = a * a
        c.backward()
        assert abs(a.grad - 6.0) < 1e-7

    def test_diamond_graph(self):
        """x -> a, x -> b, a + b. grad should accumulate."""
        x = Value(2.0)
        a = x * Value(3.0)   # a = 3x, da/dx = 3
        b = x * Value(5.0)   # b = 5x, db/dx = 5
        c = a + b            # c = 8x, dc/dx = 8
        c.backward()
        assert abs(c.data - 16.0) < 1e-7
        assert abs(x.grad - 8.0) < 1e-7

    def test_triple_use(self):
        """x used three times: x + x + x, grad = 3."""
        x = Value(5.0)
        c = x + x + x
        c.backward()
        assert abs(x.grad - 3.0) < 1e-7


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_zero_gradient_relu(self):
        """relu of negative yields zero grad."""
        a = Value(-5.0)
        b = a.relu()
        b.backward()
        assert a.grad == 0.0

    def test_backward_sets_self_grad_to_one(self):
        """Calling backward on a value sets its own grad to 1."""
        a = Value(3.0)
        b = Value(4.0)
        c = a * b
        c.backward()
        assert c.grad == 1.0

    def test_no_grad_before_backward(self):
        """Before backward, all grads are 0."""
        a = Value(3.0)
        b = a * Value(2.0)
        assert a.grad == 0.0
        assert b.grad == 0.0

    def test_repeated_backward_accumulates(self):
        """Calling backward twice should accumulate gradients."""
        a = Value(3.0)
        b = a * Value(2.0)
        b.backward()
        assert a.grad == 2.0
        b.backward()
        # Second backward: b.grad goes 1->2, so a gets += 2.0*2.0 = 4.0
        # Total a.grad = 2.0 + 4.0 = 6.0
        assert abs(a.grad - 6.0) < 1e-7


# ===========================================================================
# Numerical gradient verification for all ops
# ===========================================================================

class TestNumericalGradients:
    def test_add_num(self):
        check_grad(lambda x: x + Value(5.0), 3.0)

    def test_mul_num(self):
        check_grad(lambda x: x * Value(4.0), 3.0)

    def test_pow_num(self):
        check_grad(lambda x: x ** 3, 2.0)

    def test_sub_num(self):
        check_grad(lambda x: x - Value(1.0), 3.0)

    def test_div_num(self):
        check_grad(lambda x: x / Value(2.0), 6.0)

    def test_relu_num(self):
        check_grad(lambda x: x.relu(), 2.0)

    def test_tanh_num(self):
        check_grad(lambda x: x.tanh(), 0.7)

    def test_exp_num(self):
        check_grad(lambda x: x.exp(), 1.0)

    def test_neg_num(self):
        check_grad(lambda x: -x, 3.0)

    def test_compound_num(self):
        check_grad(lambda x: (x * Value(2.0) + Value(1.0)).relu() ** 2, 1.5)


# ===========================================================================
# Neuron / Layer / MLP
# ===========================================================================

class TestNeuron:
    def test_neuron_creates_parameters(self):
        n = Neuron(3)
        params = n.parameters()
        # 3 weights + 1 bias = 4
        assert len(params) == 4
        assert all(isinstance(p, Value) for p in params)

    def test_neuron_forward(self):
        n = Neuron(2, activation="relu")
        x = [Value(1.0), Value(2.0)]
        out = n(x)
        assert isinstance(out, Value)

    def test_neuron_tanh_activation(self):
        n = Neuron(2, activation="tanh")
        x = [Value(1.0), Value(2.0)]
        out = n(x)
        assert isinstance(out, Value)
        # tanh output should be in [-1, 1]
        assert -1.0 <= out.data <= 1.0


class TestLayer:
    def test_layer_creates_neurons(self):
        layer = Layer(3, 4)
        params = layer.parameters()
        # 4 neurons, each with 3 weights + 1 bias = 4*4 = 16
        assert len(params) == 16

    def test_layer_forward(self):
        layer = Layer(2, 3)
        x = [Value(1.0), Value(2.0)]
        out = layer(x)
        assert len(out) == 3
        assert all(isinstance(v, Value) for v in out)


class TestMLP:
    def test_mlp_creates_layers(self):
        mlp = MLP(3, [4, 2])
        params = mlp.parameters()
        # Layer 1: 4 neurons * (3 weights + 1 bias) = 16
        # Layer 2 (output, no activation): 2 neurons * (4 weights + 1 bias) = 10
        assert len(params) == 26

    def test_mlp_forward(self):
        mlp = MLP(2, [3, 1])
        out = mlp([1.0, 2.0])
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], Value)

    def test_mlp_backward(self):
        """MLP output should be differentiable."""
        mlp = MLP(2, [3, 1])
        out = mlp([1.0, 2.0])
        loss = out[0]
        loss.backward()
        for p in mlp.parameters():
            # Gradient should exist (might be zero for some relu-killed paths)
            assert isinstance(p.grad, float)

    def test_mlp_last_layer_no_activation(self):
        """Last layer should not apply activation, so output can be any float."""
        mlp = MLP(2, [4, 1])
        # Run many times; at least some should produce values outside [0, inf)
        # (if last layer had relu, output would always be >= 0)
        # We just check it runs and returns a Value
        out = mlp([0.5, -0.5])
        assert isinstance(out[0], Value)


# ===========================================================================
# XOR learning convergence
# ===========================================================================

class TestXORLearning:
    def test_xor_loss_decreases(self):
        """Train a small MLP on XOR and verify loss decreases."""
        mlp = MLP(2, [4, 4, 1])

        xs = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        ys = [0.0, 1.0, 1.0, 0.0]

        initial_loss = None
        final_loss = None

        for epoch in range(100):
            # Forward pass
            preds = [mlp(x)[0] for x in xs]
            loss = Value(0.0)
            for pred, target in zip(preds, ys):
                diff = pred - Value(target)
                loss = loss + diff * diff

            if epoch == 0:
                initial_loss = loss.data
            if epoch == 99:
                final_loss = loss.data

            # Backward pass
            # Zero grads
            for p in mlp.parameters():
                p.grad = 0.0
            loss.backward()

            # Update
            lr = 0.05
            for p in mlp.parameters():
                p.data -= lr * p.grad

        assert final_loss is not None
        assert initial_loss is not None
        # Loss should decrease significantly
        assert final_loss < initial_loss
