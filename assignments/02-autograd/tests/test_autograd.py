"""Tests for the scalar autograd engine (`Value` class).

These tests pin down the forward values and the reverse-mode gradients of
every operator and activation, verify the chain rule through multi-step
expressions, confirm gradient accumulation for shared nodes (diamond
graphs), cross-check gradients against finite differences, and exercise
edge cases (zero gradients, repeated backward calls).
"""

import math

import pytest

from autograd import Value


TOL = 1e-9


def _approx(a, b, tol=TOL):
    return abs(a - b) <= tol


def numerical_grad(f, x, h=1e-6):
    """Central finite-difference derivative of scalar f at scalar x."""
    return (f(x + h) - f(x - h)) / (2 * h)


# --------------------------------------------------------------------------
# Construction / basics
# --------------------------------------------------------------------------


def test_construction_stores_data():
    v = Value(3.0)
    assert v.data == 3.0
    assert v.grad == 0.0


def test_label_is_optional():
    v = Value(1.5, label="x")
    assert v.data == 1.5
    # label should not interfere with arithmetic
    assert (v + Value(0.5)).data == 2.0


# --------------------------------------------------------------------------
# Addition
# --------------------------------------------------------------------------


def test_add_forward():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    assert out.data == 5.0


def test_add_backward():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    out.backward()
    # d(a+b)/da = 1, d(a+b)/db = 1
    assert _approx(a.grad, 1.0)
    assert _approx(b.grad, 1.0)


def test_add_with_scalar():
    a = Value(2.0)
    out = a + 5
    assert out.data == 7.0
    out.backward()
    assert _approx(a.grad, 1.0)


def test_radd_scalar_on_left():
    a = Value(2.0)
    out = 5 + a
    assert out.data == 7.0
    out.backward()
    assert _approx(a.grad, 1.0)


# --------------------------------------------------------------------------
# Multiplication
# --------------------------------------------------------------------------


def test_mul_forward():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    assert out.data == 6.0


def test_mul_backward():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    # d(a*b)/da = b, d(a*b)/db = a
    assert _approx(a.grad, 3.0)
    assert _approx(b.grad, 2.0)


def test_mul_with_scalar():
    a = Value(4.0)
    out = a * 3
    assert out.data == 12.0
    out.backward()
    assert _approx(a.grad, 3.0)


def test_rmul_scalar_on_left():
    a = Value(4.0)
    out = 2 * a
    assert out.data == 8.0
    out.backward()
    assert _approx(a.grad, 2.0)


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def test_pow_forward():
    a = Value(3.0)
    out = a ** 2
    assert out.data == 9.0


def test_pow_backward():
    a = Value(3.0)
    out = a ** 3
    out.backward()
    # d(a**3)/da = 3 * a**2 = 27
    assert _approx(a.grad, 27.0)


def test_pow_float_exponent():
    a = Value(4.0)
    out = a ** 0.5
    assert _approx(out.data, 2.0)
    out.backward()
    # d(a**0.5)/da = 0.5 * a**-0.5 = 0.25
    assert _approx(a.grad, 0.25)


# --------------------------------------------------------------------------
# Negation / subtraction
# --------------------------------------------------------------------------


def test_neg_forward():
    a = Value(5.0)
    out = -a
    assert out.data == -5.0


def test_neg_backward():
    a = Value(5.0)
    out = -a
    out.backward()
    assert _approx(a.grad, -1.0)


def test_sub_forward():
    a, b = Value(7.0), Value(3.0)
    out = a - b
    assert out.data == 4.0


def test_sub_backward():
    a, b = Value(7.0), Value(3.0)
    out = a - b
    out.backward()
    assert _approx(a.grad, 1.0)
    assert _approx(b.grad, -1.0)


def test_sub_with_scalar():
    a = Value(7.0)
    out = a - 2
    assert out.data == 5.0
    out.backward()
    assert _approx(a.grad, 1.0)


# --------------------------------------------------------------------------
# Division
# --------------------------------------------------------------------------


def test_truediv_forward():
    a, b = Value(6.0), Value(2.0)
    out = a / b
    assert _approx(out.data, 3.0)


def test_truediv_backward():
    a, b = Value(6.0), Value(2.0)
    out = a / b
    out.backward()
    # d(a/b)/da = 1/b = 0.5 ; d(a/b)/db = -a/b**2 = -1.5
    assert _approx(a.grad, 0.5)
    assert _approx(b.grad, -1.5)


def test_truediv_with_scalar():
    a = Value(8.0)
    out = a / 4
    assert _approx(out.data, 2.0)
    out.backward()
    assert _approx(a.grad, 0.25)


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------


def test_relu_positive():
    a = Value(2.0)
    out = a.relu()
    assert out.data == 2.0
    out.backward()
    assert _approx(a.grad, 1.0)


def test_relu_negative():
    a = Value(-3.0)
    out = a.relu()
    assert out.data == 0.0
    out.backward()
    assert _approx(a.grad, 0.0)


def test_relu_zero():
    a = Value(0.0)
    out = a.relu()
    assert out.data == 0.0
    out.backward()
    # gradient at exactly 0 is conventionally 0
    assert _approx(a.grad, 0.0)


def test_tanh_forward():
    a = Value(0.5)
    out = a.tanh()
    assert _approx(out.data, math.tanh(0.5))


def test_tanh_backward():
    x = 0.7
    a = Value(x)
    out = a.tanh()
    out.backward()
    # d/dx tanh(x) = 1 - tanh(x)**2
    assert _approx(a.grad, 1 - math.tanh(x) ** 2)


def test_exp_forward():
    a = Value(1.0)
    out = a.exp()
    assert _approx(out.data, math.e)


def test_exp_backward():
    x = 0.3
    a = Value(x)
    out = a.exp()
    out.backward()
    # d/dx exp(x) = exp(x)
    assert _approx(a.grad, math.exp(x))


# --------------------------------------------------------------------------
# Chain rule through multi-step expressions
# --------------------------------------------------------------------------


def test_chain_rule_simple():
    # f = (a * b) + c
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    f = a * b + c
    f.backward()
    assert _approx(f.data, 4.0)
    assert _approx(a.grad, -3.0)  # df/da = b
    assert _approx(b.grad, 2.0)   # df/db = a
    assert _approx(c.grad, 1.0)   # df/dc = 1


def test_chain_rule_deep():
    # f = tanh(a * b + c) , a classic micrograd-style expression
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    n = a * b + c
    f = n.tanh()
    f.backward()
    assert _approx(f.data, math.tanh(n.data))
    local = 1 - math.tanh(n.data) ** 2
    assert _approx(n.grad, local)
    assert _approx(a.grad, local * b.data)
    assert _approx(b.grad, local * a.data)
    assert _approx(c.grad, local)


def test_chain_rule_with_pow_and_div():
    # f = (a ** 2) / b
    a = Value(3.0)
    b = Value(2.0)
    f = (a ** 2) / b
    f.backward()
    assert _approx(f.data, 4.5)
    # df/da = 2a/b = 3 ; df/db = -a**2/b**2 = -2.25
    assert _approx(a.grad, 3.0)
    assert _approx(b.grad, -2.25)


# --------------------------------------------------------------------------
# Gradient accumulation (diamond graphs / shared nodes)
# --------------------------------------------------------------------------


def test_accumulation_same_value_added_to_itself():
    a = Value(3.0)
    out = a + a  # = 2a
    out.backward()
    # d(2a)/da = 2 -> requires accumulation, not overwrite
    assert _approx(out.data, 6.0)
    assert _approx(a.grad, 2.0)


def test_accumulation_same_value_multiplied_by_itself():
    a = Value(3.0)
    out = a * a  # = a**2
    out.backward()
    # d(a*a)/da = 2a = 6
    assert _approx(out.data, 9.0)
    assert _approx(a.grad, 6.0)


def test_accumulation_diamond_graph():
    # a feeds two branches that recombine: f = (a * b) + (a * c)
    a = Value(4.0)
    b = Value(2.0)
    c = Value(3.0)
    f = a * b + a * c
    f.backward()
    # df/da = b + c = 5
    assert _approx(a.grad, 5.0)
    assert _approx(b.grad, 4.0)
    assert _approx(c.grad, 4.0)


def test_accumulation_reused_three_times():
    a = Value(2.0)
    f = a + a + a  # = 3a
    f.backward()
    assert _approx(f.data, 6.0)
    assert _approx(a.grad, 3.0)


# --------------------------------------------------------------------------
# Numerical gradient verification (finite differences)
# --------------------------------------------------------------------------


def test_numerical_check_polynomial():
    # f(x) = 3x**2 + 2x + 1, df/dx = 6x + 2
    def build(xval):
        x = Value(xval)
        return 3 * (x ** 2) + 2 * x + 1, x

    out, x = build(1.5)
    out.backward()

    def f(xval):
        return build(xval)[0].data

    assert _approx(x.grad, numerical_grad(f, 1.5), tol=1e-5)


def test_numerical_check_with_activation():
    # f(x) = tanh(2x + 1)
    def f(xval):
        x = Value(xval)
        return (2 * x + 1).tanh().data

    x = Value(0.4)
    out = (2 * x + 1).tanh()
    out.backward()
    assert _approx(x.grad, numerical_grad(f, 0.4), tol=1e-5)


def test_numerical_check_division_and_exp():
    # f(x) = exp(x) / (x + 2)
    def f(xval):
        x = Value(xval)
        return (x.exp() / (x + 2)).data

    x = Value(0.8)
    out = x.exp() / (x + 2)
    out.backward()
    assert _approx(x.grad, numerical_grad(f, 0.8), tol=1e-5)

@pytest.mark.parametrize("xval", [-1.3, -0.2, 0.0, 0.5, 2.1])
def test_numerical_check_relu_sweep(xval):
    # relu has a kink at 0; skip the exact-zero point for finite differences
    def f(v):
        return Value(v).relu().data

    x = Value(xval)
    out = x.relu()
    out.backward()
    if xval == 0.0:
        assert _approx(x.grad, 0.0)
    else:
        assert _approx(x.grad, numerical_grad(f, xval), tol=1e-5)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_grad_starts_at_zero():
    a = Value(1.0)
    assert a.grad == 0.0


def test_backward_seeds_output_grad_to_one():
    a = Value(5.0)
    out = a * 1
    out.backward()
    assert _approx(out.grad, 1.0)


def test_zero_gradient_for_unused_input():
    # b does not influence out, so its gradient stays 0
    a = Value(2.0)
    b = Value(9.0)
    out = a * 3
    out.backward()
    assert _approx(a.grad, 3.0)
    assert _approx(b.grad, 0.0)


def test_constant_expression_has_no_input_grad():
    a = Value(4.0)
    out = a - a  # = 0, grad cancels to 0 via accumulation
    out.backward()
    assert _approx(out.data, 0.0)
    assert _approx(a.grad, 0.0)


def test_repeated_backward_resets_seed_but_accumulates_input_grads():
    # backward() hard-assigns the output seed (out.grad = 1) each call, so
    # the seed does not accumulate. Input grads ARE never zeroed, so calling
    # backward() twice (no manual zeroing) adds the same per-call delta
    # again, doubling the input gradients.
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    g1_a, g1_b = a.grad, b.grad
    out.backward()
    assert _approx(a.grad, 2 * g1_a)
    assert _approx(b.grad, 2 * g1_b)


def test_topological_order_handles_long_chain():
    # Build a long chain to ensure topo sort visits every node once.
    x = Value(1.0)
    acc = x
    for _ in range(20):
        acc = acc + x  # acc = (n+1) * x after n iterations
    acc.backward()
    # acc = 21 * x  -> d/dx = 21
    assert _approx(acc.data, 21.0)
    assert _approx(x.grad, 21.0)


def test_pow_zero_exponent_backward_no_zero_division():
    # d(x**0)/dx == 0 for all x, including x == 0. The backward pass must not
    # raise ZeroDivisionError by computing 0 ** -1.
    x = Value(0.0)
    out = x ** 0
    assert out.data == 1.0
    out.backward()  # would raise ZeroDivisionError without the other==0 guard
    assert _approx(x.grad, 0.0)


def test_backward_handles_deep_chain_without_recursion_error():
    # A linear chain far deeper than Python's default recursion limit (~1000)
    # must back-propagate via the iterative topo sort without RecursionError.
    x = Value(1.0)
    acc = x
    depth = 3000
    for _ in range(depth):
        acc = acc + x  # acc = (depth + 1) * x
    acc.backward()
    assert _approx(acc.data, float(depth + 1))
    assert _approx(x.grad, float(depth + 1))
