"""Tests for the bytecode compiler + stack VM (Assignment 3).

The convenience entry point is:

    from src.lang import execute
    execute(source: str) -> list   # list of values passed to `print`

`print` records the value (not its string form): a printed integer comes
back as an int, a printed string as a str, booleans as bools, nil as None.
"""
import pytest

from src.lang import execute


# Arithmetic + operator precedence

def test_integer_addition():
    assert execute("print 1 + 2;") == [3]


def test_subtraction_and_negatives():
    assert execute("print 10 - 4;") == [6]
    assert execute("print 3 - 10;") == [-7]


def test_multiplication():
    assert execute("print 6 * 7;") == [42]


def test_integer_division_is_truncating():
    assert execute("print 7 / 2;") == [3]


def test_float_division():
    assert execute("print 7.0 / 2;") == [3.5]


def test_modulo():
    assert execute("print 17 % 5;") == [2]


def test_precedence_mul_before_add():
    assert execute("print 2 + 3 * 4;") == [14]


def test_precedence_with_parens():
    assert execute("print (2 + 3) * 4;") == [20]


def test_unary_negation():
    assert execute("print -5 + 8;") == [3]


def test_nested_unary():
    assert execute("print --5;") == [5]


def test_mixed_int_float_promotes_to_float():
    assert execute("print 1 + 2.5;") == [3.5]


def test_chained_arithmetic_left_assoc():
    assert execute("print 20 - 5 - 3;") == [12]


# Variables + scoping

def test_let_and_read():
    assert execute("let x = 5; print x;") == [5]


def test_reassignment():
    assert execute("let x = 1; x = x + 9; print x;") == [10]


def test_multiple_variables():
    assert execute("let a = 2; let b = 3; print a * b;") == [6]


def test_assignment_expression_value():
    assert execute("let x = 0; print (x = 7);") == [7]


def test_undefined_variable_raises():
    with pytest.raises(Exception):
        execute("print y;")


def test_function_reads_outer_variable():
    src = """
    let base = 100;
    fn add(n) { return base + n; }
    print add(5);
    """
    assert execute(src) == [105]


def test_function_local_scope_does_not_leak():
    src = """
    fn f() { let secret = 42; return secret; }
    print f();
    """
    assert execute(src) == [42]
    with pytest.raises(Exception):
        execute("fn f() { let secret = 42; return secret; } f(); print secret;")


def test_parameter_shadows_outer():
    src = """
    let n = 1;
    fn f(n) { return n * 10; }
    print f(5);
    print n;
    """
    assert execute(src) == [50, 1]


# Control flow

def test_if_true_branch():
    assert execute("if 1 < 2 { print 1; } else { print 0; }") == [1]


def test_if_false_branch():
    assert execute("if 5 < 2 { print 1; } else { print 0; }") == [0]


def test_if_without_else_skips():
    assert execute("if false { print 1; } print 2;") == [2]


def test_while_loop_counts():
    src = """
    let i = 0;
    while i < 3 {
        print i;
        i = i + 1;
    }
    """
    assert execute(src) == [0, 1, 2]


def test_while_sum():
    src = """
    let i = 1;
    let total = 0;
    while i <= 5 {
        total = total + i;
        i = i + 1;
    }
    print total;
    """
    assert execute(src) == [15]


def test_nested_blocks_and_loops():
    src = """
    let i = 0;
    while i < 2 {
        let j = 0;
        while j < 2 {
            print i * 10 + j;
            j = j + 1;
        }
        i = i + 1;
    }
    """
    assert execute(src) == [0, 1, 10, 11]


def test_nested_if_in_while():
    src = """
    let i = 0;
    while i < 4 {
        if i % 2 == 0 { print i; }
        i = i + 1;
    }
    """
    assert execute(src) == [0, 2]


# Functions, recursion

def test_simple_function_call():
    src = """
    fn square(x) { return x * x; }
    print square(9);
    """
    assert execute(src) == [81]


def test_function_no_args():
    src = """
    fn answer() { return 42; }
    print answer();
    """
    assert execute(src) == [42]


def test_function_multiple_args():
    src = """
    fn add3(a, b, c) { return a + b + c; }
    print add3(1, 2, 3);
    """
    assert execute(src) == [6]


def test_recursion_factorial():
    src = """
    fn fact(n) {
        if n <= 1 { return 1; }
        return n * fact(n - 1);
    }
    print fact(5);
    """
    assert execute(src) == [120]


def test_recursion_fibonacci():
    src = """
    fn fib(n) {
        if n < 2 { return n; }
        return fib(n - 1) + fib(n - 2);
    }
    print fib(10);
    """
    assert execute(src) == [55]


def test_return_without_value_is_nil():
    src = """
    fn noop() { return; }
    print noop();
    """
    assert execute(src) == [None]


def test_function_with_loop_inside():
    src = """
    fn sumto(n) {
        let i = 1;
        let s = 0;
        while i <= n {
            s = s + i;
            i = i + 1;
        }
        return s;
    }
    print sumto(10);
    """
    assert execute(src) == [55]


def test_mutual_call_chain():
    src = """
    fn inc(x) { return x + 1; }
    fn double(x) { return x * 2; }
    print double(inc(4));
    """
    assert execute(src) == [10]


# Strings

def test_string_literal():
    assert execute("print \"hello\";") == ["hello"]


def test_string_concatenation():
    assert execute("print \"foo\" + \"bar\";") == ["foobar"]


def test_string_concat_with_variable():
    assert execute("let name = \"world\"; print \"hello \" + name;") == ["hello world"]


def test_string_equality():
    assert execute("print \"a\" == \"a\";") == [True]
    assert execute("print \"a\" == \"b\";") == [False]


# Boolean logic

def test_boolean_literals():
    assert execute("print true;") == [True]
    assert execute("print false;") == [False]


def test_comparisons():
    assert execute("print 3 < 5;") == [True]
    assert execute("print 5 < 3;") == [False]
    assert execute("print 5 == 5;") == [True]
    assert execute("print 5 != 5;") == [False]
    assert execute("print 5 >= 5;") == [True]
    assert execute("print 4 > 5;") == [False]
    assert execute("print 5 <= 6;") == [True]


def test_logical_and():
    assert execute("print true and true;") == [True]
    assert execute("print true and false;") == [False]


def test_logical_or():
    assert execute("print false or true;") == [True]
    assert execute("print false or false;") == [False]


def test_not():
    assert execute("print not true;") == [False]
    assert execute("print not false;") == [True]


def test_and_short_circuits():
    assert execute("print false and undefined_thing;") == [False]


def test_or_short_circuits():
    assert execute("print true or undefined_thing;") == [True]


def test_boolean_combination():
    assert execute("print (3 < 5) and (10 > 2);") == [True]


# Errors

def test_division_by_zero_raises():
    with pytest.raises(Exception):
        execute("print 1 / 0;")


def test_modulo_by_zero_raises():
    with pytest.raises(Exception):
        execute("print 1 % 0;")


def test_undefined_variable_in_expression():
    with pytest.raises(Exception):
        execute("print 1 + nope;")


def test_type_error_add_number_and_string():
    with pytest.raises(Exception):
        execute("print 1 + \"x\";")


def test_calling_undefined_function():
    with pytest.raises(Exception):
        execute("print missing(1);")


# Complex programs

def test_fizzbuzz():
    src = """
    let i = 1;
    while i <= 15 {
        if i % 15 == 0 {
            print \"FizzBuzz\";
        } else {
            if i % 3 == 0 {
                print \"Fizz\";
            } else {
                if i % 5 == 0 {
                    print \"Buzz\";
                } else {
                    print i;
                }
            }
        }
        i = i + 1;
    }
    """
    expected = [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"]
    assert execute(src) == expected


def test_gcd():
    src = """
    fn gcd(a, b) {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        return a;
    }
    print gcd(48, 36);
    print gcd(17, 5);
    """
    assert execute(src) == [12, 1]


def test_bubble_sort_via_swaps():
    src = """
    let a = 3;
    let b = 1;
    let c = 2;
    if a > b {
        let t = a;
        a = b;
        b = t;
    }
    if b > c {
        let t = b;
        b = c;
        c = t;
    }
    if a > b {
        let t = a;
        a = b;
        b = t;
    }
    print a;
    print b;
    print c;
    """
    assert execute(src) == [1, 2, 3]


def test_power_function_recursive():
    src = """
    fn power(base, exp) {
        if exp == 0 { return 1; }
        return base * power(base, exp - 1);
    }
    print power(2, 10);
    """
    assert execute(src) == [1024]


def test_counter_accumulation_program():
    src = """
    fn is_even(n) { return n % 2 == 0; }
    let i = 0;
    let evens = 0;
    while i < 10 {
        if is_even(i) { evens = evens + 1; }
        i = i + 1;
    }
    print evens;
    """
    assert execute(src) == [5]
