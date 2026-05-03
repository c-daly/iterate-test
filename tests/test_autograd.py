"""Tests for scalar autograd engine and tiny neural net library."""
import math
import random

import pytest

from src.autograd import Value
from src.nn import Neuron, Layer, MLP


# ---------------------------------------------------------------------------
# Value: forward arithmetic
# ---------------------------------------------------------------------------

class TestValueForward:
    def test_init(self):
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0

    def test_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_pow(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_neg(self):
        a = Value(3.0)
        c = -a
        assert c.data == -3.0

    def test_sub(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_truediv(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        assert c.data == 3.0

    def test_radd(self):
        a = Value(3.0)
        c = 2 + a
        assert c.data == 5.0

    def test_rmul(self):
        a = Value(3.0)
        c = 2 * a
        assert c.data == 6.0

    def test_add_with_scalar(self):
        a = Value(3.0)
        c = a + 2
        assert c.data == 5.0

    def test_mul_with_scalar(self):
        a = Value(3.0)
        c = a * 2
        assert c.data == 6.0


# ---------------------------------------------------------------------------
# Activations: forward
# ---------------------------------------------------------------------------

class TestActivationsForward:
    def test_relu_pos(self):
        a = Value(2.0)
        assert a.relu().data == 2.0

    def test_relu_neg(self):
        a = Value(-2.0)
        assert a.relu().data == 0.0

    def test_relu_zero(self):
        a = Value(0.0)
        assert a.relu().data == 0.0

    def test_tanh(self):
        a = Value(0.5)
        assert a.tanh().data == pytest.approx(math.tanh(0.5))

    def test_exp(self):
        a = Value(1.0)
        assert a.exp().data == pytest.approx(math.e)


# ---------------------------------------------------------------------------
# Backward: per-op gradients
# ---------------------------------------------------------------------------

class TestBackwardPerOp:
    def test_grad_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_grad_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_grad_pow(self):
        a = Value(2.0)
        c = a ** 3
        c.backward()
        assert a.grad == pytest.approx(12.0)

    def test_grad_neg(self):
        a = Value(3.0)
        c = -a
        c.backward()
        assert a.grad == -1.0

    def test_grad_sub(self):
        a = Value(5.0)
        b = Value(2.0)
        c = a - b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_grad_div(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        c.backward()
        assert a.grad == pytest.approx(0.5)
        assert b.grad == pytest.approx(-1.5)

    def test_grad_relu_pos(self):
        a = Value(2.0)
        c = a.relu()
        c.backward()
        assert a.grad == 1.0

    def test_grad_relu_neg(self):
        a = Value(-2.0)
        c = a.relu()
        c.backward()
        assert a.grad == 0.0

    def test_grad_tanh(self):
        x = 0.5
        a = Value(x)
        c = a.tanh()
        c.backward()
        expected = 1.0 - math.tanh(x) ** 2
        assert a.grad == pytest.approx(expected)

    def test_grad_exp(self):
        x = 1.5
        a = Value(x)
        c = a.exp()
        c.backward()
        assert a.grad == pytest.approx(math.exp(x))


# ---------------------------------------------------------------------------
# Chain rule + accumulation + topo
# ---------------------------------------------------------------------------

class TestChainAndAccumulation:
    def test_chain_rule(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        d = Value(5.0)
        out = (a * b + c) * d
        out.backward()
        # df/da = b*d, df/db = a*d, df/dc = d, df/dd = a*b+c
        assert a.grad == pytest.approx(15.0)
        assert b.grad == pytest.approx(10.0)
        assert c.grad == pytest.approx(5.0)
        assert d.grad == pytest.approx(10.0)

    def test_gradient_accumulation_diamond(self):
        a = Value(3.0)
        c = a * a
        c.backward()
        assert a.grad == pytest.approx(6.0)

    def test_gradient_accumulation_three_uses(self):
        a = Value(2.0)
        c = a + a + a
        c.backward()
        assert a.grad == pytest.approx(3.0)

    def test_gradient_accumulation_mixed(self):
        a = Value(3.0)
        b = Value(4.0)
        c = a * b + a
        c.backward()
        assert a.grad == pytest.approx(5.0)
        assert b.grad == pytest.approx(3.0)

    def test_repeated_backward_accumulates(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        first_a = a.grad
        c.backward()
        assert a.grad == pytest.approx(first_a * 2)

    def test_zero_gradient_branch(self):
        a = Value(2.0)
        c = (-a).relu()
        c.backward()
        assert a.grad == 0.0


# ---------------------------------------------------------------------------
# Finite difference numerical-gradient verification
# ---------------------------------------------------------------------------

def _numerical_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


class TestNumericalGrad:
    def test_polynomial(self):
        def f(x):
            return (x * x + 3 * x + 2) * math.tanh(x)

        x_val = 0.7
        x = Value(x_val)
        out = (x * x + 3 * x + 2) * x.tanh()
        out.backward()
        assert x.grad == pytest.approx(_numerical_grad(f, x_val), rel=1e-4)

    def test_exp_chain(self):
        def f(x):
            return math.exp(x * x + 1)

        x_val = 0.5
        x = Value(x_val)
        out = (x * x + 1).exp()
        out.backward()
        assert x.grad == pytest.approx(_numerical_grad(f, x_val), rel=1e-4)

    def test_div_pow(self):
        def f(x):
            return ((x ** 3) + 2) / (x + 1)

        x_val = 1.3
        x = Value(x_val)
        out = (x ** 3 + 2) / (x + 1)
        out.backward()
        assert x.grad == pytest.approx(_numerical_grad(f, x_val), rel=1e-4)


# ---------------------------------------------------------------------------
# Neuron / Layer / MLP
# ---------------------------------------------------------------------------

class TestNeuron:
    def test_forward_returns_value(self):
        random.seed(0)
        n = Neuron(3)
        out = n([Value(1.0), Value(2.0), Value(3.0)])
        assert isinstance(out, Value)

    def test_relu_neuron_nonneg(self):
        random.seed(0)
        n = Neuron(3, activation="relu")
        out = n([Value(1.0), Value(2.0), Value(3.0)])
        assert out.data >= 0.0

    def test_tanh_neuron_in_range(self):
        random.seed(0)
        n = Neuron(3, activation="tanh")
        out = n([Value(1.0), Value(2.0), Value(3.0)])
        assert -1.0 <= out.data <= 1.0

    def test_linear_neuron_no_activation(self):
        random.seed(0)
        n = Neuron(3, activation="none")
        out = n([Value(1.0), Value(2.0), Value(3.0)])
        assert isinstance(out, Value)

    def test_neuron_parameters_count(self):
        n = Neuron(4)
        assert len(n.parameters()) == 5


class TestLayer:
    def test_layer_output_size(self):
        random.seed(0)
        layer = Layer(3, 5)
        out = layer([Value(1.0), Value(2.0), Value(3.0)])
        assert len(out) == 5
        assert all(isinstance(o, Value) for o in out)

    def test_layer_parameters(self):
        layer = Layer(3, 5)
        assert len(layer.parameters()) == 20


class TestMLP:
    def test_mlp_forward_shape(self):
        random.seed(0)
        mlp = MLP(3, [4, 4, 1])
        out = mlp([1.0, 2.0, 3.0])
        assert len(out) == 1
        assert isinstance(out[0], Value)

    def test_mlp_parameters_count(self):
        # 4*(3+1) + 4*(4+1) + 1*(4+1) = 16 + 20 + 5 = 41
        mlp = MLP(3, [4, 4, 1])
        assert len(mlp.parameters()) == 41

    def test_mlp_last_layer_no_activation(self):
        random.seed(123)
        seen_negative = False
        for _ in range(50):
            m = MLP(3, [4, 1])
            x = [random.uniform(-1, 1) for _ in range(3)]
            out = m(x)
            if out[0].data < 0:
                seen_negative = True
                break
        assert seen_negative, "Last layer must be linear -> negative outputs reachable"


# ---------------------------------------------------------------------------
# XOR convergence training
# ---------------------------------------------------------------------------

class TestXORConvergence:
    def test_xor_loss_decreases(self):
        random.seed(1337)
        mlp = MLP(2, [8, 8, 1])
        data = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ]

        def total_loss():
            loss = Value(0.0)
            for x, y in data:
                pred = mlp(x)[0]
                diff = pred - y
                loss = loss + diff * diff
            return loss

        for p in mlp.parameters():
            p.grad = 0.0
        initial = total_loss()
        initial_data = initial.data

        lr = 0.05
        for step in range(200):
            for p in mlp.parameters():
                p.grad = 0.0
            loss = total_loss()
            loss.backward()
            for p in mlp.parameters():
                p.data -= lr * p.grad

        for p in mlp.parameters():
            p.grad = 0.0
        final = total_loss()
        assert final.data < initial_data, (
            f"loss did not decrease: initial={initial_data}, final={final.data}"
        )
        assert final.data < 0.5, f"final XOR loss too high: {final.data}"
