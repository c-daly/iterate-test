"""Test suite for the compiler + stack VM (Assignment 03).

Covers lexer, parser, compiler, VM, and the high-level execute convenience.
Every spec bullet has at least one dedicated test.
"""
import pytest

from src.compiler import Chunk, Compiler, OpCode
from src.lang import execute
from src.lexer import Lexer, TokenType
from src.parser import Parser
from src.vm import VM

NL = chr(10)


class TestLexer:
    def test_single_number(self):
        toks = Lexer("42").tokenize()
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 42

    def test_float(self):
        toks = Lexer("3.14").tokenize()
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 3.14

    def test_string_literal(self):
        toks = Lexer(chr(34) + "hello" + chr(34)).tokenize()
        assert toks[0].type == TokenType.STRING
        assert toks[0].value == "hello"

    def test_keywords(self):
        src = "let if else while fn return print true false nil and or not"
        toks = Lexer(src).tokenize()
        kinds = [t.type for t in toks if t.type != TokenType.EOF]
        expected = [
            TokenType.LET, TokenType.IF, TokenType.ELSE, TokenType.WHILE,
            TokenType.FN, TokenType.RETURN, TokenType.PRINT, TokenType.TRUE,
            TokenType.FALSE, TokenType.NIL, TokenType.AND, TokenType.OR,
            TokenType.NOT,
        ]
        assert kinds == expected

    def test_two_char_operators(self):
        toks = Lexer("== != <= >=").tokenize()
        kinds = [t.type for t in toks if t.type != TokenType.EOF]
        assert kinds == [TokenType.EQ, TokenType.NE, TokenType.LE, TokenType.GE]

    def test_line_tracking(self):
        toks = Lexer("1" + NL + "2" + NL + "3").tokenize()
        assert toks[0].line == 1
        assert toks[1].line == 2
        assert toks[2].line == 3

    def test_skip_whitespace_and_comments(self):
        toks = Lexer("  // comment" + NL + "  42  ").tokenize()
        non_eof = [t for t in toks if t.type != TokenType.EOF]
        assert len(non_eof) == 1
        assert non_eof[0].value == 42

    def test_identifiers(self):
        toks = Lexer("foo bar_baz qux1").tokenize()
        idents = [t.value for t in toks if t.type == TokenType.IDENT]
        assert idents == ["foo", "bar_baz", "qux1"]


class TestParser:
    def test_let_statement(self):
        ast = Parser(Lexer("let x = 1;").tokenize()).parse()
        assert ast is not None
        assert len(ast.statements) == 1

    def test_if_else(self):
        src = "if 1 { print 1; } else { print 2; }"
        ast = Parser(Lexer(src).tokenize()).parse()
        assert len(ast.statements) == 1

    def test_function_decl(self):
        src = "fn add(a, b) { return a + b; }"
        ast = Parser(Lexer(src).tokenize()).parse()
        assert len(ast.statements) == 1


class TestArithmetic:
    def test_add(self):
        assert execute("print 1 + 2;") == ["3"]

    def test_sub(self):
        assert execute("print 5 - 2;") == ["3"]

    def test_mul(self):
        assert execute("print 3 * 4;") == ["12"]

    def test_div(self):
        assert execute("print 10 / 2;") == ["5"]

    def test_mod(self):
        assert execute("print 10 % 3;") == ["1"]

    def test_neg(self):
        assert execute("print -5;") == ["-5"]

    def test_double_neg(self):
        assert execute("print --5;") == ["5"]

    def test_precedence_mul_over_add(self):
        assert execute("print 2 + 3 * 4;") == ["14"]

    def test_precedence_paren(self):
        assert execute("print (2 + 3) * 4;") == ["20"]

    def test_precedence_unary(self):
        assert execute("print -2 * 3;") == ["-6"]

    def test_float_arith(self):
        assert execute("print 1.5 + 2.5;") == ["4.0"]

    def test_left_associative_sub(self):
        assert execute("print 10 - 3 - 2;") == ["5"]


class TestComparison:
    def test_eq(self):
        assert execute("print 1 == 1;") == ["true"]
        assert execute("print 1 == 2;") == ["false"]

    def test_ne(self):
        assert execute("print 1 != 2;") == ["true"]

    def test_lt_gt(self):
        assert execute("print 1 < 2;") == ["true"]
        assert execute("print 1 > 2;") == ["false"]

    def test_le_ge(self):
        assert execute("print 2 <= 2;") == ["true"]
        assert execute("print 2 >= 2;") == ["true"]

    def test_not(self):
        assert execute("print not true;") == ["false"]
        assert execute("print not false;") == ["true"]

    def test_and_basic(self):
        assert execute("print true and true;") == ["true"]
        assert execute("print true and false;") == ["false"]

    def test_or_basic(self):
        assert execute("print true or false;") == ["true"]
        assert execute("print false or false;") == ["false"]

    def test_and_or_precedence(self):
        assert execute("print true or false and false;") == ["true"]


class TestShortCircuit:
    def test_and_short_circuits_division_by_zero(self):
        out = execute("if false and (1 / 0 == 0) { print 1; } else { print 2; }")
        assert out == ["2"]

    def test_or_short_circuits_division_by_zero(self):
        out = execute("if true or (1 / 0 == 0) { print 1; } else { print 2; }")
        assert out == ["1"]

    def test_and_short_circuit_no_side_effect(self):
        src = (
            "let counter = 0; "
            "fn bump() { counter = counter + 1; return true; } "
            "if false and bump() { print 1; } "
            "print counter;"
        )
        assert execute(src) == ["0"]

    def test_or_short_circuit_no_side_effect(self):
        src = (
            "let counter = 0; "
            "fn bump() { counter = counter + 1; return true; } "
            "if true or bump() { print 1; } "
            "print counter;"
        )
        assert execute(src) == ["1", "0"]


class TestVariables:
    def test_let_and_use(self):
        assert execute("let x = 5; print x;") == ["5"]

    def test_assignment(self):
        assert execute("let x = 1; x = 2; print x;") == ["2"]

    def test_function_reads_outer(self):
        src = (
            "let x = 10; "
            "fn read() { return x; } "
            "print read();"
        )
        assert execute(src) == ["10"]

    def test_function_local_does_not_leak(self):
        src = (
            "fn f() { let local = 99; return local; } "
            "let result = f(); "
            "print result;"
        )
        assert execute(src) == ["99"]

    def test_undefined_variable_raises(self):
        with pytest.raises(Exception):
            execute("print missing;")


class TestControlFlow:
    def test_if_true(self):
        assert execute("if true { print 1; }") == ["1"]

    def test_if_false(self):
        assert execute("if false { print 1; }") == []

    def test_if_else(self):
        assert execute("if false { print 1; } else { print 2; }") == ["2"]

    def test_while(self):
        src = "let i = 0; while i < 3 { print i; i = i + 1; }"
        assert execute(src) == ["0", "1", "2"]

    def test_if_in_while(self):
        src = (
            "let i = 0; "
            "while i < 5 { "
            "if i % 2 == 0 { print i; } "
            "i = i + 1; "
            "}"
        )
        assert execute(src) == ["0", "2", "4"]

    def test_while_in_while(self):
        src = (
            "let i = 0; "
            "while i < 2 { "
            "let j = 0; "
            "while j < 2 { print i * 10 + j; j = j + 1; } "
            "i = i + 1; "
            "}"
        )
        assert execute(src) == ["0", "1", "10", "11"]


class TestFunctions:
    def test_simple_call(self):
        src = "fn add(a, b) { return a + b; } print add(2, 3);"
        assert execute(src) == ["5"]

    def test_no_arg_function(self):
        src = "fn give() { return 42; } print give();"
        assert execute(src) == ["42"]

    def test_factorial(self):
        src = (
            "fn fact(n) { "
            "if n <= 1 { return 1; } "
            "return n * fact(n - 1); "
            "} "
            "print fact(5);"
        )
        assert execute(src) == ["120"]

    def test_fibonacci(self):
        src = (
            "fn fib(n) { "
            "if n < 2 { return n; } "
            "return fib(n - 1) + fib(n - 2); "
            "} "
            "print fib(10);"
        )
        assert execute(src) == ["55"]

    def test_function_no_explicit_return(self):
        src = "fn nothing() {} nothing();"
        assert execute(src) == []


class TestStrings:
    def test_string_print(self):
        assert execute("print " + chr(34) + "hello" + chr(34) + ";") == ["hello"]

    def test_string_concat(self):
        src = "print " + chr(34) + "foo" + chr(34) + " + " + chr(34) + "bar" + chr(34) + ";"
        assert execute(src) == ["foobar"]

    def test_string_eq(self):
        src = "print " + chr(34) + "x" + chr(34) + " == " + chr(34) + "x" + chr(34) + ";"
        assert execute(src) == ["true"]


class TestErrors:
    def test_division_by_zero(self):
        with pytest.raises(Exception):
            execute("print 1 / 0;")

    def test_mod_by_zero(self):
        with pytest.raises(Exception):
            execute("print 5 % 0;")

    def test_type_error_add(self):
        src = "print 1 + " + chr(34) + "x" + chr(34) + ";"
        with pytest.raises(Exception):
            execute(src)

    def test_undefined_variable(self):
        with pytest.raises(Exception):
            execute("print y;")


class TestComplexPrograms:
    def test_fizzbuzz(self):
        q = chr(34)
        src = (
            "let i = 1; "
            "while i <= 15 { "
            "if i % 15 == 0 { print " + q + "FizzBuzz" + q + "; } "
            "else { "
            "if i % 3 == 0 { print " + q + "Fizz" + q + "; } "
            "else { "
            "if i % 5 == 0 { print " + q + "Buzz" + q + "; } "
            "else { print i; } "
            "} "
            "} "
            "i = i + 1; "
            "}"
        )
        out = execute(src)
        expected = [
            "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8",
            "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz",
        ]
        assert out == expected

    def test_gcd(self):
        src = (
            "fn gcd(a, b) { "
            "while b != 0 { "
            "let t = b; "
            "b = a % b; "
            "a = t; "
            "} "
            "return a; "
            "} "
            "print gcd(48, 18); "
            "print gcd(100, 75);"
        )
        assert execute(src) == ["6", "25"]

    def test_iterative_factorial(self):
        src = (
            "fn fact(n) { "
            "let acc = 1; "
            "let i = 2; "
            "while i <= n { "
            "acc = acc * i; "
            "i = i + 1; "
            "} "
            "return acc; "
            "} "
            "print fact(6);"
        )
        assert execute(src) == ["720"]


class TestCompilerEmission:
    def _compile(self, src: str) -> Chunk:
        ast = Parser(Lexer(src).tokenize()).parse()
        return Compiler().compile(ast)

    def test_and_emits_jump_if_false(self):
        chunk = self._compile("print true and false;")
        ops = [instr[0] for instr in chunk.code]
        assert OpCode.JUMP_IF_FALSE in ops

    def test_or_emits_jump(self):
        chunk = self._compile("print true or false;")
        ops = [instr[0] for instr in chunk.code]
        assert OpCode.JUMP in ops or OpCode.JUMP_IF_FALSE in ops

    def test_halt_at_end(self):
        chunk = self._compile("print 1;")
        assert chunk.code[-1][0] == OpCode.HALT


class TestVM:
    def test_vm_runs_const_print_halt(self):
        chunk = Chunk()
        idx = chunk.add_const(7)
        chunk.emit(OpCode.CONST, idx)
        chunk.emit(OpCode.PRINT)
        chunk.emit(OpCode.HALT)
        out = VM().run(chunk)
        assert out == ["7"]


class TestLexicalScoping:
    """Regression tests for static (lexical) scoping of free variables.

    Functions must look up free variables in the environment where they
    were *defined*, not where they are *called*. The previous (buggy)
    behaviour chained the new call frame off the caller env, giving
    dynamic scoping.
    """

    def test_function_does_not_see_callers_local(self):
        # The classic dynamic-vs-static scoping discriminator. Under
        # dynamic scoping `f` would see `caller`s local `x` (= 99);
        # under lexical scoping it sees the outer `x` (= 10) from
        # where `f` was defined.
        src = (
            "let x = 10; "
            "fn f() { return x; } "
            "fn caller() { let x = 99; return f(); } "
            "print caller();"
        )
        assert execute(src) == ["10"]

    def test_function_sees_outer_var_after_caller_mutation(self):
        # Free-variable resolution still goes through the env chain
        # at lookup time, so subsequent assignments to the outer `x`
        # ARE visible (this is normal lexical scoping with mutable
        # bindings, not snapshot capture).
        src = (
            "let x = 1; "
            "fn read_x() { return x; } "
            "x = 2; "
            "print read_x();"
        )
        assert execute(src) == ["2"]

    def test_returned_inner_fn_keeps_lexical_env(self):
        # `outer` declares a local `n`, then declares `inner` which
        # references `n`. `outer` returns `inner` itself (not the
        # result of calling it). When the returned closure is called
        # from top-level (where `n` is *not* in scope), lexical
        # scoping must still find `n` via inner.def_env. Under the
        # old dynamic-scoping bug this raises `Undefined variable: n`.
        src = (
            "fn outer() { "
            "let n = 7; "
            "fn inner() { return n; } "
            "return inner; "
            "} "
            "let g = outer(); "
            "print g();"
        )
        assert execute(src) == ["7"]
