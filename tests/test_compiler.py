"""End-to-end and per-layer tests for the bytecode compiler + VM."""
import pytest

from src.lexer import tokenize, LexError
from src.parser import parse, ParseError
from src.vm import run, VMError
from src.lang import execute


NL = chr(10)


def src(*lines):
    return NL.join(lines)


# ---------- Lexer ----------------------------------------------------------

class TestLexer:
    def test_numbers_int_and_float(self):
        toks = tokenize("42 3.14")
        assert toks[0].type == "NUMBER" and toks[0].value == 42
        assert toks[1].type == "NUMBER" and toks[1].value == 3.14

    def test_keywords_vs_idents(self):
        toks = tokenize("let x return foo if")
        kinds = [t.type for t in toks[:-1]]
        assert kinds == ["let", "IDENT", "return", "IDENT", "if"]

    def test_operators_and_punct(self):
        toks = tokenize("== != <= >= = + - * / % ( ) { } , ;")
        types = [t.type for t in toks[:-1]]
        assert types == ["==", "!=", "<=", ">=", "=", "+", "-", "*", "/", "%",
                         "(", ")", "{", "}", ",", ";"]

    def test_string_literal(self):
        toks = tokenize(chr(34) + "hello world" + chr(34))
        assert toks[0].type == "STRING" and toks[0].value == "hello world"

    def test_string_escapes(self):
        # \n inside quotes
        s = chr(34) + chr(92) + "n" + chr(34)
        toks = tokenize(s)
        assert toks[0].value == chr(10)

    def test_comments_skipped(self):
        toks = tokenize("let x = 1; // a comment" + NL + "print x;")
        types = [t.type for t in toks]
        assert "//" not in types
        assert "comment" not in types

    def test_lex_error_on_bad_char(self):
        with pytest.raises(LexError):
            tokenize("let x = @;")


# ---------- Parser ---------------------------------------------------------

class TestParser:
    def test_parse_let(self):
        prog = parse(tokenize("let x = 1;"))
        assert len(prog.statements) == 1
        assert prog.statements[0].name == "x"

    def test_precedence_addition_vs_multiply(self):
        prog = parse(tokenize("let x = 1 + 2 * 3;"))
        let = prog.statements[0]
        # outer must be Binary +
        assert let.value.op == "+"
        # right side must be Binary *
        assert let.value.right.op == "*"

    def test_parse_error_missing_semicolon(self):
        with pytest.raises(ParseError):
            parse(tokenize("let x = 1"))

    def test_parse_function_with_params(self):
        prog = parse(tokenize("fn add(a, b) { return a + b; }"))
        fn = prog.statements[0]
        assert fn.name == "add"
        assert fn.params == ["a", "b"]


# ---------- VM (small, programmatic) --------------------------------------

class TestVMDirect:
    def test_const_print_halt(self):
        from src.compiler import CodeObject
        co = CodeObject(name="<t>", instructions=[
            ("CONST", 7), ("PRINT",), ("HALT",),
        ])
        assert run(co) == ["7"]

    def test_arithmetic_ops(self):
        from src.compiler import CodeObject
        co = CodeObject(name="<t>", instructions=[
            ("CONST", 10), ("CONST", 3), ("ADD",), ("PRINT",),
            ("CONST", 10), ("CONST", 3), ("SUB",), ("PRINT",),
            ("CONST", 10), ("CONST", 3), ("MUL",), ("PRINT",),
            ("CONST", 10), ("CONST", 5), ("DIV",), ("PRINT",),
            ("CONST", 10), ("CONST", 3), ("MOD",), ("PRINT",),
            ("HALT",),
        ])
        assert run(co) == ["13", "7", "30", "2", "1"]


# ---------- End-to-end programs -------------------------------------------

class TestExecute:
    def test_arithmetic_precedence(self):
        assert execute("print 2 + 3 * 4;") == ["14"]
        assert execute("print (2 + 3) * 4;") == ["20"]
        assert execute("print 10 - 3 - 2;") == ["5"]
        assert execute("print 20 / 4 / 5;") == ["1"]
        assert execute("print -3 + 5;") == ["2"]
        assert execute("print 7 % 4;") == ["3"]

    def test_let_and_assignment(self):
        out = execute(src(
            "let x = 5;",
            "x = x + 1;",
            "print x;",
        ))
        assert out == ["6"]

    def test_scoping_block_can_read_outer(self):
        out = execute(src(
            "let x = 1;",
            "if true { print x; }",
        ))
        assert out == ["1"]

    def test_if_else(self):
        assert execute("if 1 < 2 { print 100; } else { print 200; }") == ["100"]
        assert execute("if 1 > 2 { print 100; } else { print 200; }") == ["200"]
        assert execute("if true { print 1; }") == ["1"]
        assert execute("if false { print 1; }") == []

    def test_while_loop(self):
        out = execute(src(
            "let i = 0;",
            "while i < 5 { print i; i = i + 1; }",
        ))
        assert out == ["0", "1", "2", "3", "4"]

    def test_function_decl_and_call(self):
        out = execute(src(
            "fn add(a, b) { return a + b; }",
            "print add(2, 3);",
        ))
        assert out == ["5"]

    def test_recursion_factorial(self):
        out = execute(src(
            "fn fact(n) { if n <= 1 { return 1; } return n * fact(n - 1); }",
            "print fact(6);",
        ))
        assert out == ["720"]

    def test_recursion_fibonacci(self):
        out = execute(src(
            "fn fib(n) { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }",
            "print fib(10);",
        ))
        assert out == ["55"]

    def test_nested_function_reads_outer(self):
        out = execute(src(
            "let x = 100;",
            "fn show() { print x; }",
            "show();",
        ))
        assert out == ["100"]

    def test_strings_concat(self):
        Q = chr(34)
        prog = "print " + Q + "foo" + Q + " + " + Q + "bar" + Q + ";"
        assert execute(prog) == ["foobar"]

    def test_booleans_and_short_circuit(self):
        assert execute("print true and false;") == ["false"]
        assert execute("print true and true;") == ["true"]
        assert execute("print false or true;") == ["true"]
        assert execute("print not true;") == ["false"]
        assert execute("print not false;") == ["true"]

    def test_short_circuit_does_not_evaluate_right(self):
        # rhs assignment would change y; if short-circuited, y stays 0
        out = execute(src(
            "let y = 0;",
            "fn bump() { y = y + 1; return true; }",
            "let r = false and bump();",
            "print y;",
            "r = true or bump();",
            "print y;",
        ))
        assert out == ["0", "0"]

    def test_undefined_variable_raises(self):
        with pytest.raises(VMError):
            execute("print not_defined;")

    def test_division_by_zero_raises(self):
        with pytest.raises(VMError):
            execute("print 1 / 0;")

    def test_modulo_by_zero_raises(self):
        with pytest.raises(VMError):
            execute("print 1 % 0;")

    def test_fizzbuzz(self):
        out = execute(src(
            "let i = 1;",
            "while i <= 15 {",
            "    if i % 15 == 0 { print " + chr(34) + "FizzBuzz" + chr(34) + "; }",
            "    else { if i % 3 == 0 { print " + chr(34) + "Fizz" + chr(34) + "; }",
            "    else { if i % 5 == 0 { print " + chr(34) + "Buzz" + chr(34) + "; }",
            "    else { print i; } } }",
            "    i = i + 1;",
            "}",
        ))
        expected = ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz",
                    "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]
        assert out == expected

    def test_gcd(self):
        out = execute(src(
            "fn gcd(a, b) { while b != 0 { let t = b; b = a % b; a = t; } return a; }",
            "print gcd(48, 18);",
            "print gcd(100, 75);",
            "print gcd(17, 5);",
        ))
        assert out == ["6", "25", "1"]

    def test_sort_three_via_swaps(self):
        # Bubble-sort-style ordering of three named variables.
        out = execute(src(
            "let a = 3; let b = 1; let c = 2;",
            "if a > b { let t = a; a = b; b = t; }",
            "if b > c { let t = b; b = c; c = t; }",
            "if a > b { let t = a; a = b; b = t; }",
            "print a; print b; print c;",
        ))
        assert out == ["1", "2", "3"]

    def test_nested_blocks_and_assignment_chain(self):
        out = execute(src(
            "let x = 0;",
            "if true {",
            "    if true { x = x + 1; }",
            "    x = x + 10;",
            "}",
            "print x;",
        ))
        assert out == ["11"]

    def test_function_local_does_not_leak(self):
        # writing inside a function to a not-yet-declared name creates local
        out = execute(src(
            "let x = 5;",
            "fn f() { let x = 99; print x; }",
            "f();",
            "print x;",
        ))
        assert out == ["99", "5"]
