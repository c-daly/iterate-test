import math
import random
import pytest
from autograd import Value
from nn import Neuron, Layer, MLP


def test_add():
    a = Value(2)
    b = Value(3)
    c = a + b
    c.backward()
    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul():
    a = Value(2)
    b = Value(3)
    c = a * b
    c.backward()
    assert c.data == 6.0
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_pow():
    a = Value(3)
    c = a ** 2
    c.backward()
    assert c.data == 9.0
    assert a.grad == 6.0


def test_neg_sub_div():
    a = Value(6)
    b = Value(3)

    # neg
    neg_a = -a
    assert neg_a.data == -6.0

    # sub
    d = a - b
    assert d.data == 3.0
    d.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0

    # div (fresh values to avoid grad accumulation)
    a2 = Value(6)
    b2 = Value(3)
    e = a2 / b2
    assert e.data == pytest.approx(2.0)
    e.backward()
    # d(a/b)/da = 1/b = 1/3
    assert a2.grad == pytest.approx(1.0 / 3.0)
    # d(a/b)/db = -a/b^2 = -6/9 = -2/3
    assert b2.grad == pytest.approx(-6.0 / 9.0)


def test_radd_rmul():
    a = Value(3)
    b = 2 + a
    assert b.data == 5.0
    b.backward()
    assert a.grad == 1.0

    c = Value(3)
    d = 2 * c
    assert d.data == 6.0
    d.backward()
    assert c.grad == 2.0


def test_relu():
    a = Value(2)
    b = a.relu()
    assert b.data == 2.0
    b.backward()
    assert a.grad == 1.0

    c = Value(-2)
    d = c.relu()
    assert d.data == 0.0
    d.backward()
    assert c.grad == 0.0


def test_tanh():
    a = Value(0.5)
    b = a.tanh()
    expected = math.tanh(0.5)
    assert b.data == pytest.approx(expected)
    assert -1.0 <= b.data <= 1.0
    b.backward()
    # d(tanh)/dx = 1 - tanh^2(x)
    expected_grad = 1 - expected ** 2
    assert a.grad == pytest.approx(expected_grad)


def test_exp():
    a = Value(1)
    b = a.exp()
    assert b.data == pytest.approx(math.e)
    b.backward()
    # d(exp(x))/dx = exp(x)
    assert a.grad == pytest.approx(math.e)


def test_chain_rule():
    a = Value(2)
    b = Value(3)
    c = Value(1)
    # expr = (a*b + c)**2
    expr = (a * b + c) ** 2
    expr.backward()
    # (a*b + c) = 7, expr = 49
    assert expr.data == 49.0
    # d/da = 2*(a*b+c) * b = 2*7*3 = 42
    assert a.grad == pytest.approx(42.0)
    # d/db = 2*(a*b+c) * a = 2*7*2 = 28
    assert b.grad == pytest.approx(28.0)
    # d/dc = 2*(a*b+c) * 1 = 14
    assert c.grad == pytest.approx(14.0)


def test_gradient_accumulation():
    a = Value(3)
    c = a * a
    c.backward()
    # d(a^2)/da = 2a = 6
    assert a.grad == pytest.approx(6.0)


def test_diamond_graph():
    a = Value(2)
    b = Value(3)
    c = Value(4)
    d = a * b + a * c
    d.backward()
    # d/da = b + c = 3 + 4 = 7 (accumulates both paths)
    assert a.grad == pytest.approx(7.0)
    # d/db = a = 2
    assert b.grad == pytest.approx(2.0)
    # d/dc = a = 2
    assert c.grad == pytest.approx(2.0)


def test_numerical_gradient():
    """Finite difference verification."""
    eps = 1e-7
    tol = 1e-5

    def f(x_val, y_val):
        x = Value(x_val)
        y = Value(y_val)
        out = (x * y + x.relu()).tanh()
        return out, x, y

    # Compute analytical gradients
    out, x, y = f(2.0, 3.0)
    out.backward()
    anal_dx = x.grad
    anal_dy = y.grad

    # Numerical gradient for x
    out_plus, _, _ = f(2.0 + eps, 3.0)
    out_minus, _, _ = f(2.0 - eps, 3.0)
    num_dx = (out_plus.data - out_minus.data) / (2 * eps)

    # Numerical gradient for y
    out_plus, _, _ = f(2.0, 3.0 + eps)
    out_minus, _, _ = f(2.0, 3.0 - eps)
    num_dy = (out_plus.data - out_minus.data) / (2 * eps)

    assert anal_dx == pytest.approx(num_dx, abs=tol)
    assert anal_dy == pytest.approx(num_dy, abs=tol)


def test_zero_grad():
    a = Value(5)
    b = Value(3)
    c = a * b
    # Before backward, grads should be 0
    assert a.grad == 0.0
    assert b.grad == 0.0
    assert c.grad == 0.0


def test_repeated_backward():
    a = Value(2)
    b = Value(3)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(3.0)
    assert b.grad == pytest.approx(2.0)
    # Second backward — grads should double (accumulate)
    c.backward()
    assert a.grad == pytest.approx(6.0)
    assert b.grad == pytest.approx(4.0)



# --- Neural Network Tests ---

def test_neuron_forward():
    random.seed(42)
    n = Neuron(3)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    out = n(x)
    assert isinstance(out, Value)


def test_neuron_parameters():
    n = Neuron(3)
    assert len(n.parameters()) == 4  # 3 weights + 1 bias


def test_layer_forward():
    random.seed(42)
    layer = Layer(3, 4)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    out = layer(x)
    assert len(out) == 4
    assert all(isinstance(v, Value) for v in out)


def test_layer_parameters():
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 4 * (3 + 1)  # 4 neurons * (3 weights + 1 bias)


def test_mlp_forward():
    random.seed(42)
    mlp = MLP(3, [4, 4, 1])
    out = mlp([1.0, 2.0, 3.0])
    assert len(out) == 1
    assert isinstance(out[0], Value)


def test_mlp_parameters():
    mlp = MLP(3, [4, 4, 1])
    params = mlp.parameters()
    expected = 4*(3+1) + 4*(4+1) + 1*(4+1)  # 16 + 20 + 5 = 41
    assert len(params) == expected


def test_xor_learning():
    random.seed(3)
    mlp = MLP(2, [4, 1])
    xs = [[0,0],[0,1],[1,0],[1,1]]
    ys = [0, 1, 1, 0]

    for step in range(100):
        # Forward
        preds = [mlp(x)[0] for x in xs]
        loss = sum((p - y)**2 for p, y in zip(preds, ys))

        # Zero grad
        for p in mlp.parameters():
            p.grad = 0.0

        # Backward
        loss.backward()

        # Update
        for p in mlp.parameters():
            p.data -= 0.05 * p.grad

    # Final loss should be small
    preds = [mlp(x)[0] for x in xs]
    final_loss = sum((p.data - y)**2 for p, y in zip(preds, ys))
    assert final_loss < 0.1
