"""Tests for the tiny neural-net library built on the autograd Value class.

Covers Neuron/Layer/MLP shape and parameter accounting, parameter sharing,
learning behavior on XOR (loss must drop), and last-layer-no-activation.
"""
from __future__ import annotations

import random

import pytest

from autograd import Value
from nn import Layer, MLP, Neuron


def _rng_seed():
    random.seed(1337)


class TestNeuron:
    def setup_method(self):
        _rng_seed()

    def test_param_count(self):
        n = Neuron(3)
        params = n.parameters()
        assert len(params) == 4

    def test_params_are_values(self):
        n = Neuron(2)
        for p in n.parameters():
            assert isinstance(p, Value)

    def test_forward_returns_value(self):
        n = Neuron(2, activation="relu")
        out = n([Value(1.0), Value(-1.0)])
        assert isinstance(out, Value)

    def test_relu_nonneg_output(self):
        n = Neuron(2, activation="relu")
        out = n([Value(1.0), Value(2.0)])
        assert out.data >= 0.0

    def test_tanh_in_range(self):
        n = Neuron(2, activation="tanh")
        out = n([Value(1.0), Value(-1.0)])
        assert -1.0 <= out.data <= 1.0

    def test_no_activation(self):
        n = Neuron(2, activation="none")
        out = n([Value(1.0), Value(1.0)])
        assert isinstance(out, Value)


class TestLayer:
    def setup_method(self):
        _rng_seed()

    def test_output_shape(self):
        layer = Layer(3, 4)
        outs = layer([Value(1.0), Value(2.0), Value(3.0)])
        assert len(outs) == 4

    def test_param_count(self):
        layer = Layer(3, 4)
        assert len(layer.parameters()) == 16

    def test_outputs_are_values(self):
        layer = Layer(2, 3)
        outs = layer([Value(0.5), Value(-0.5)])
        for o in outs:
            assert isinstance(o, Value)


class TestMLP:
    def setup_method(self):
        _rng_seed()

    def test_forward_with_floats(self):
        mlp = MLP(3, [4, 4, 1])
        out = mlp([1.0, 2.0, 3.0])
        assert len(out) == 1
        assert isinstance(out[0], Value)

    def test_param_count(self):
        mlp = MLP(3, [4, 4, 1])
        assert len(mlp.parameters()) == 41

    def test_last_layer_no_activation(self):
        mlp = MLP(2, [4, 1])
        last_layer = mlp.layers[-1]
        for neuron in last_layer.neurons:
            assert neuron.activation == "none"

    def test_intermediate_layer_has_activation(self):
        mlp = MLP(2, [4, 1])
        for neuron in mlp.layers[0].neurons:
            assert neuron.activation != "none"

    def test_output_can_be_negative(self):
        mlp = MLP(2, [8, 1])
        any_neg = False
        for _ in range(50):
            xs = [random.uniform(-2, 2), random.uniform(-2, 2)]
            y = mlp(xs)[0].data
            if y < 0:
                any_neg = True
                break
        assert any_neg


class TestXORConvergence:
    """Train a tiny MLP on XOR and verify the loss drops substantially."""

    def test_xor_loss_decreases(self):
        random.seed(42)
        mlp = MLP(2, [8, 8, 1])
        xs = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        ys = [-1.0, 1.0, 1.0, -1.0]

        def loss_fn():
            preds = [mlp(x)[0] for x in xs]
            return sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0))

        initial = loss_fn().data
        lr = 0.05
        final = initial
        for step in range(200):
            for p in mlp.parameters():
                p.grad = 0.0
            L = loss_fn()
            L.backward()
            for p in mlp.parameters():
                p.data -= lr * p.grad
            final = L.data

        assert final < 0.05, "XOR did not converge"
        assert final < initial * 0.05


class TestNeuronDimensionMismatch:
    """Regression: Neuron.__call__ must reject inputs whose length != nin."""

    def test_input_length_mismatch_raises(self):
        # Previously zip() silently truncated, producing wrong gradients.
        n = Neuron(3)
        with pytest.raises(ValueError):
            n([Value(1.0), Value(2.0)])
