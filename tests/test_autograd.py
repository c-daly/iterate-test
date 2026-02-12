"""Tests for scalar autograd engine and neural network modules."""

import math
import pytest
from src.autograd import Value
from src.nn import Neuron, Layer, MLP


class TestValueCreation:
    def test_basic_creation(self):
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0

    def test_creation_with_label(self):
        v = Value(3.0, label="x")
        assert v.label == "x"

    def test_default_label(self):
        v = Value(3.0)
        assert v.label == ""


class TestAddition:
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

    def test_radd_float_plus_value(self):
        a = Value(2.0)
        c = 3.0 + a
        assert c.data == 5.0

    def test_radd_int_plus_value(self):
        a = Value(2.0)
        c = 3 + a
        assert c.data == 5.0

    def test_add_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0


class TestMultiplication:
    def test_mul_two_values(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_mul_value_and_float(self):
        a = Value(2.0)
        c = a * 3.0
        assert c.data == 6.0

    def test_mul_value_and_int(self):
        a = Value(2.0)
        c = a * 3
        assert c.data == 6.0

    def test_rmul_float_times_value(self):
        a = Value(2.0)
        c = 3.0 * a
        assert c.data == 6.0

    def test_rmul_int_times_value(self):
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


class TestPower:
    def test_pow_int(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_pow_float(self):
        a = Value(4.0)
        c = a ** 0.5
        assert abs(c.data - 2.0) < 1e-9

    def test_pow_gradient(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert abs(a.grad - 6.0) < 1e-9

    def test_pow_gradient_cube(self):
        a = Value(2.0)
        c = a ** 3
        c.backward()
        assert abs(a.grad - 12.0) < 1e-9


class TestNegation:
    def test_neg(self):
        a = Value(3.0)
        c = -a
        assert c.data == -3.0

    def test_neg_gradient(self):
        a = Value(3.0)
        c = -a
        c.backward()
        assert a.grad == -1.0


class TestSubtraction:
    def test_sub_two_values(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_sub_value_and_float(self):
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


class TestDivision:
    def test_div_two_values(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-9

    def test_div_value_and_float(self):
        a = Value(6.0)
        c = a / 3.0
        assert abs(c.data - 2.0) < 1e-9

    def test_div_gradient(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        c.backward()
        assert abs(a.grad - 1.0 / 3.0) < 1e-9
        assert abs(b.grad - (-6.0 / 9.0)) < 1e-9


class TestActivations:
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

    def test_tanh(self):
        a = Value(0.5)
        c = a.tanh()
        assert abs(c.data - math.tanh(0.5)) < 1e-9

    def test_tanh_gradient(self):
        a = Value(0.5)
        c = a.tanh()
        c.backward()
        expected = 1 - math.tanh(0.5) ** 2
        assert abs(a.grad - expected) < 1e-9

    def test_exp(self):
        a = Value(1.0)
        c = a.exp()
        assert abs(c.data - math.e) < 1e-9

    def test_exp_gradient(self):
        a = Value(1.0)
        c = a.exp()
        c.backward()
        assert abs(a.grad - math.e) < 1e-9


class TestChainRule:
    def test_chain_add_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        d = (a + b) * c
        d.backward()
        assert a.grad == 4.0
        assert b.grad == 4.0
        assert c.grad == 5.0

    def test_chain_mul_pow(self):
        a = Value(2.0)
        b = Value(3.0)
        c = (a * b) ** 2
        c.backward()
        assert abs(a.grad - 36.0) < 1e-9
        assert abs(b.grad - 24.0) < 1e-9

    def test_complex_expression(self):
        a = Value(1.0)
        b = Value(2.0)
        c = Value(-1.0)
        d = (a * b + c).tanh()
        d.backward()
        dtanh = 1 - math.tanh(1.0) ** 2
        assert abs(a.grad - b.data * dtanh) < 1e-9
        assert abs(b.grad - a.data * dtanh) < 1e-9
        assert abs(c.grad - dtanh) < 1e-9


class TestGradientAccumulation:
    def test_diamond_graph(self):
        a = Value(3.0)
        c = a * a
        c.backward()
        assert abs(a.grad - 6.0) < 1e-9

    def test_shared_node(self):
        a = Value(2.0)
        b = a + 1
        c = a + 2
        d = b * c
        d.backward()
        assert abs(a.grad - 7.0) < 1e-9

    def test_triple_use(self):
        a = Value(2.0)
        c = a * a * a
        c.backward()
        assert abs(a.grad - 12.0) < 1e-9


class TestNumericalGradient:
    def _numerical_grad(self, f, x, h=1e-5):
        x.data += h
        fph = f().data
        x.data -= 2 * h
        fmh = f().data
        x.data += h
        return (fph - fmh) / (2 * h)

    def test_add_numerical(self):
        a = Value(2.0)
        b = Value(3.0)
        f = lambda: a + b
        c = f()
        c.backward()
        assert abs(a.grad - self._numerical_grad(f, a)) < 1e-4
        assert abs(b.grad - self._numerical_grad(f, b)) < 1e-4

    def test_mul_numerical(self):
        a = Value(2.0)
        b = Value(3.0)
        f = lambda: a * b
        c = f()
        c.backward()
        assert abs(a.grad - self._numerical_grad(f, a)) < 1e-4
        assert abs(b.grad - self._numerical_grad(f, b)) < 1e-4

    def test_pow_numerical(self):
        a = Value(3.0)
        f = lambda: a ** 3
        c = f()
        c.backward()
        assert abs(a.grad - self._numerical_grad(f, a)) < 1e-4

    def test_tanh_numerical(self):
        a = Value(0.7)
        f = lambda: a.tanh()
        c = f()
        c.backward()
        assert abs(a.grad - self._numerical_grad(f, a)) < 1e-4

    def test_exp_numerical(self):
        a = Value(1.5)
        f = lambda: a.exp()
        c = f()
        c.backward()
        assert abs(a.grad - self._numerical_grad(f, a)) < 1e-4

    def test_complex_expression_numerical(self):
        a = Value(1.5)
        b = Value(-2.0)
        f = lambda: ((a * b + Value(1.0)).tanh()) * a
        c = f()
        c.backward()
        assert abs(a.grad - self._numerical_grad(f, a)) < 1e-4
        a = Value(1.5)
        b = Value(-2.0)
        f2 = lambda: ((a * b + Value(1.0)).tanh()) * a
        c2 = f2()
        c2.backward()
        assert abs(b.grad - self._numerical_grad(f2, b)) < 1e-4


class TestEdgeCases:
    def test_zero_gradient_initial(self):
        a = Value(3.0)
        assert a.grad == 0.0

    def test_backward_on_leaf(self):
        a = Value(3.0)
        a.backward()
        assert a.grad == 1.0

    def test_add_zero(self):
        a = Value(3.0)
        c = a + 0
        c.backward()
        assert a.grad == 1.0

    def test_mul_zero(self):
        a = Value(3.0)
        c = a * 0
        assert c.data == 0.0
        c.backward()
        assert a.grad == 0.0

    def test_mul_one(self):
        a = Value(3.0)
        c = a * 1
        assert c.data == 3.0
        c.backward()
        assert a.grad == 1.0

    def test_pow_zero(self):
        a = Value(3.0)
        c = a ** 0
        assert c.data == 1.0

    def test_pow_one(self):
        a = Value(3.0)
        c = a ** 1
        c.backward()
        assert abs(a.grad - 1.0) < 1e-9


class TestNeuron:
    def test_neuron_creation(self):
        n = Neuron(3)
        params = n.parameters()
        assert len(params) == 4

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
        assert -1.0 <= out.data <= 1.0

    def test_neuron_no_activation(self):
        n = Neuron(2, activation="")
        x = [Value(1.0), Value(2.0)]
        out = n(x)
        assert isinstance(out, Value)

    def test_neuron_parameters_are_values(self):
        n = Neuron(3)
        for p in n.parameters():
            assert isinstance(p, Value)


class TestLayer:
    def test_layer_creation(self):
        layer = Layer(3, 4)
        params = layer.parameters()
        assert len(params) == 4 * (3 + 1)

    def test_layer_forward(self):
        layer = Layer(2, 3)
        x = [Value(1.0), Value(2.0)]
        out = layer(x)
        assert len(out) == 3
        for o in out:
            assert isinstance(o, Value)


class TestMLP:
    def test_mlp_creation(self):
        mlp = MLP(2, [4, 3, 1])
        params = mlp.parameters()
        assert len(params) == 12 + 15 + 4

    def test_mlp_forward(self):
        mlp = MLP(2, [4, 1])
        out = mlp([1.0, 2.0])
        assert len(out) == 1
        assert isinstance(out[0], Value)

    def test_mlp_last_layer_no_activation(self):
        mlp = MLP(1, [2, 1])
        last_layer = mlp.layers[-1]
        for neuron in last_layer.neurons:
            for w in neuron.w:
                w.data = -5.0
            neuron.b.data = -5.0
        out = mlp([1.0])
        assert out[0].data < 0

    def test_mlp_gradient_flow(self):
        mlp = MLP(2, [3, 1])
        out = mlp([1.0, 2.0])
        loss = out[0]
        loss.backward()
        params = mlp.parameters()
        has_nonzero = any(abs(p.grad) > 1e-10 for p in params)
        assert has_nonzero


class TestXORLearning:
    def test_xor_convergence(self):
        import random
        random.seed(42)
        mlp = MLP(2, [8, 8, 1])
        xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        ys = [0.0, 1.0, 1.0, 0.0]
        initial_loss = None
        lr = 0.05
        for epoch in range(300):
            predictions = [mlp(x)[0] for x in xs]
            loss = sum((pred - yt) ** 2 for pred, yt in zip(predictions, ys))
            if epoch == 0:
                initial_loss = loss.data
            for p in mlp.parameters():
                p.grad = 0.0
            loss.backward()
            for p in mlp.parameters():
                p.data -= lr * p.grad
        final_loss = loss.data
        assert final_loss < initial_loss
        assert final_loss < 0.5


class TestRepr:
    def test_repr(self):
        v = Value(3.14)
        r = repr(v)
        assert "3.14" in r
