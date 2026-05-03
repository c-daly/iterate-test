"""Tests for the scalar autograd Value class.

Covers individual ops with analytic gradient checks, activations, chain rule,
diamond gradient accumulation, finite-difference verification, reverse ops,
and repeated-backward semantics.
"""
from __future__ import annotations

import math
from typing import Callable

import pytest

from autograd import Value


def finite_diff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Central finite-difference derivative of f at x."""
    return (f(x + h) - f(x - h)) / (2 * h)


def approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class TestConstruction:
    def test_data_stored(self):
        v = Value(3.0)
        assert v.data == 3.0

    def test_label_default_empty(self):
        v = Value(1.0)
        assert v.label == ""

    def test_label_custom(self):
        v = Value(2.0, label="x")
        assert v.label == "x"

    def test_grad_initialized_zero(self):
        v = Value(5.0)
        assert v.grad == 0.0

    def test_repr_contains_data(self):
        v = Value(7.5, label="abc")
        s = repr(v)
        assert "7.5" in s


class TestAdd:
    def test_forward(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_grad(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_add_scalar_right(self):
        a = Value(2.0)
        c = a + 4.0
        assert c.data == 6.0
        c.backward()
        assert a.grad == 1.0

    def test_radd_scalar(self):
        a = Value(2.0)
        c = 4.0 + a
        assert c.data == 6.0
        c.backward()
        assert a.grad == 1.0

    def test_radd_int(self):
        a = Value(2.0)
        c = 5 + a
        assert c.data == 7.0


class TestMul:
    def test_forward(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_grad(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_mul_scalar(self):
        a = Value(2.0)
        c = a * 5.0
        assert c.data == 10.0
        c.backward()
        assert a.grad == 5.0

    def test_rmul_scalar(self):
        a = Value(3.0)
        c = 2.0 * a
        assert c.data == 6.0
        c.backward()
        assert a.grad == 2.0

    def test_rmul_int(self):
        a = Value(3.0)
        c = 2 * a
        assert c.data == 6.0
        c.backward()
        assert a.grad == 2.0


class TestPow:
    def test_forward(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_grad_int_exponent(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert a.grad == 6.0

    def test_grad_float_exponent(self):
        a = Value(2.0)
        c = a ** 3.0
        c.backward()
        assert a.grad == pytest.approx(12.0)

    def test_pow_finite_diff(self):
        x_val = 1.7
        a = Value(x_val)
        c = a ** 4
        c.backward()
        expected = finite_diff(lambda x: x ** 4, x_val)
        assert approx(a.grad, expected)


class TestNeg:
    def test_forward(self):
        a = Value(3.0)
        c = -a
        assert c.data == -3.0

    def test_grad(self):
        a = Value(3.0)
        c = -a
        c.backward()
        assert a.grad == -1.0


class TestSub:
    def test_forward(self):
        a = Value(5.0)
        b = Value(2.0)
        c = a - b
        assert c.data == 3.0

    def test_grad(self):
        a = Value(5.0)
        b = Value(2.0)
        c = a - b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_sub_scalar(self):
        a = Value(5.0)
        c = a - 2.0
        assert c.data == 3.0
        c.backward()
        assert a.grad == 1.0

    def test_rsub_scalar(self):
        a = Value(2.0)
        c = 5.0 - a
        assert c.data == 3.0
        c.backward()
        assert a.grad == -1.0


class TestDiv:
    def test_forward(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        assert c.data == pytest.approx(3.0)

    def test_grad(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        c.backward()
        assert a.grad == pytest.approx(0.5)
        assert b.grad == pytest.approx(-1.5)

    def test_div_scalar(self):
        a = Value(6.0)
        c = a / 2.0
        assert c.data == pytest.approx(3.0)
        c.backward()
        assert a.grad == pytest.approx(0.5)

    def test_rdiv_scalar(self):
        a = Value(2.0)
        c = 6.0 / a
        assert c.data == pytest.approx(3.0)
        c.backward()
        assert a.grad == pytest.approx(-1.5)


class TestRelu:
    def test_positive(self):
        a = Value(3.0)
        c = a.relu()
        assert c.data == 3.0
        c.backward()
        assert a.grad == 1.0

    def test_negative(self):
        a = Value(-2.0)
        c = a.relu()
        assert c.data == 0.0
        c.backward()
        assert a.grad == 0.0

    def test_zero(self):
        a = Value(0.0)
        c = a.relu()
        assert c.data == 0.0
        c.backward()
        assert a.grad in (0.0, 1.0)


class TestTanh:
    def test_forward(self):
        a = Value(0.5)
        c = a.tanh()
        assert c.data == pytest.approx(math.tanh(0.5))

    def test_grad(self):
        a = Value(0.5)
        c = a.tanh()
        c.backward()
        expected = 1 - math.tanh(0.5) ** 2
        assert a.grad == pytest.approx(expected)

    def test_finite_diff(self):
        x = 0.7
        a = Value(x)
        c = a.tanh()
        c.backward()
        expected = finite_diff(math.tanh, x)
        assert approx(a.grad, expected)


class TestExp:
    def test_forward(self):
        a = Value(1.0)
        c = a.exp()
        assert c.data == pytest.approx(math.e)

    def test_grad(self):
        a = Value(1.5)
        c = a.exp()
        c.backward()
        assert a.grad == pytest.approx(math.exp(1.5))

    def test_finite_diff(self):
        x = 0.3
        a = Value(x)
        c = a.exp()
        c.backward()
        expected = finite_diff(math.exp, x)
        assert approx(a.grad, expected)


class TestChainRule:
    def test_two_step(self):
        a = Value(4.0)
        c = (a + 2) * 3
        c.backward()
        assert a.grad == 3.0

    def test_polynomial(self):
        x = Value(2.0)
        y = x ** 3 + 2 * x
        y.backward()
        assert x.grad == pytest.approx(14.0)

    def test_finite_diff_polynomial(self):
        x_val = 1.3
        x = Value(x_val)
        y = x ** 3 + 2 * x
        y.backward()
        expected = finite_diff(lambda v: v ** 3 + 2 * v, x_val)
        assert approx(x.grad, expected)

    def test_composite_with_activation(self):
        x_val = 0.4
        x = Value(x_val)
        y = (2 * x + 1).tanh()
        y.backward()
        expected = finite_diff(lambda v: math.tanh(2 * v + 1), x_val)
        assert approx(x.grad, expected)


class TestDiamondAccumulation:
    def test_diamond_a_times_b_plus_a_times_c(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        out = a * b + a * c
        out.backward()
        assert a.grad == 7.0
        assert b.grad == 2.0
        assert c.grad == 2.0

    def test_self_addition(self):
        a = Value(3.0)
        out = a + a
        out.backward()
        assert a.grad == 2.0

    def test_self_multiplication(self):
        a = Value(4.0)
        out = a * a
        out.backward()
        assert a.grad == 8.0

    def test_deeper_diamond(self):
        a = Value(2.0)
        out = (a + 1) * (a + 2)
        out.backward()
        assert a.grad == pytest.approx(7.0)


class TestBackwardEdgeCases:
    def test_backward_on_leaf(self):
        a = Value(3.0)
        a.backward()
        assert a.grad == 1.0

    def test_repeated_backward_accumulates(self):
        a = Value(2.0)
        b = Value(3.0)
        out = a * b
        out.backward()
        first_a = a.grad
        out.backward()
        # Repeated backward accumulates additively into both leaf and intermediate
        # node grads (no auto-zero), so a.grad strictly grows.
        assert a.grad > first_a

    def test_zero_gradient_branch(self):
        a = Value(-1.0)
        b = Value(2.0)
        out = a.relu() + b
        out.backward()
        assert a.grad == 0.0
        assert b.grad == 1.0

    def test_complex_finite_diff_check(self):
        x_val = 1.5

        def f(v: float) -> float:
            relu_arg = v * 2 + 1
            r = relu_arg if relu_arg > 0 else 0.0
            return r * (v + 3)

        x = Value(x_val)
        out = (x * 2 + 1).relu() * (x + 3)
        out.backward()
        expected = finite_diff(f, x_val)
        assert approx(x.grad, expected, tol=1e-3)


class TestPowZeroExponentRegression:
    """Regression: Value(0.0) ** 0 must not raise ZeroDivisionError on backward."""

    def test_zero_pow_zero_backward_no_raise(self):
        # Value(0.0) ** 0 used to ZeroDivisionError in _backward via
        # 0 * 0**-1; the derivative of x**0 is 0 everywhere defined.
        a = Value(0.0)
        c = a ** 0
        assert c.data == 1.0  # Python: 0.0 ** 0 == 1.0
        c.backward()
        assert a.grad == 0.0


class TestDeepGraphBackwardRegression:
    """Regression: backward() must handle graphs deeper than Python recursion limit."""

    def test_deep_chain_backward_no_recursion_error(self):
        # Recursive topological sort blew Python recursion limit on long
        # chains; the iterative DFS must handle depth >> sys.getrecursionlimit.
        import sys
        depth = 2000
        assert depth > sys.getrecursionlimit() // 2
        x = Value(0.0)
        cur = x
        for _ in range(depth):
            cur = cur + 1
        assert cur.data == float(depth)
        cur.backward()
        assert x.grad == 1.0
