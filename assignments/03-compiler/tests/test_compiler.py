"""Comprehensive tests for the bytecode compiler and stack VM."""

import sys
import os
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lexer import tokenize, Token, TokenType
from src.parser import (
    parse,
    Program,
    NumberLit,
    StringLit,
    BoolLit,
    NilLit,
    Identifier,
    UnaryOp,
    BinaryOp,
    Assignment,
    CallExpr,
    LetStmt,
    PrintStmt,
    ExprStmt,
    Block,
    IfStmt,
    WhileStmt,
    FnDecl,
    ReturnStmt,
)
from src.compiler import compile_ast, OpCode, CompiledProgram
from src.vm import run, VMError
from src.lang import execute


# ============================================================
# Lexer tests
# ============================================================


class TestLexer:
    def test_empty_source(self):
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_integer(self):
        tokens = tokenize("42")
        tok = tokens[0]
        assert isinstance(tok, Token)
        assert tok.type == TokenType.NUMBER
        assert tok.value == 42

    def test_float(self):
        tokens = tokenize("3.14")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 3.14

    def test_string_double_quotes(self):
        tokens = tokenize('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_identifiers(self):
        tokens = tokenize("foo bar_baz x1")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "foo"
        assert tokens[1].type == TokenType.IDENT
        assert tokens[1].value == "bar_baz"
        assert tokens[2].type == TokenType.IDENT
        assert tokens[2].value == "x1"

    def test_keywords(self):
        src = "let if else while print fn return true false nil and or not"
        tokens = tokenize(src)
        expected = [
            TokenType.LET, TokenType.IF, TokenType.ELSE,
            TokenType.WHILE, TokenType.PRINT, TokenType.FN,
            TokenType.RETURN, TokenType.TRUE, TokenType.FALSE,
            TokenType.NIL, TokenType.AND, TokenType.OR, TokenType.NOT,
        ]
        for tok, exp in zip(tokens, expected):
            assert tok.type == exp

    def test_operators(self):
        src = "+ - * / % == != < > <= >= ="
        tokens = tokenize(src)
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
            TokenType.SLASH, TokenType.PERCENT, TokenType.EQ,
            TokenType.NE, TokenType.LT, TokenType.GT,
            TokenType.LE, TokenType.GE, TokenType.ASSIGN,
        ]
        for tok, exp in zip(tokens, expected):
            assert tok.type == exp

    def test_delimiters(self):
        src = "( ) { } ; ,"
        tokens = tokenize(src)
        expected = [
            TokenType.LPAREN, TokenType.RPAREN,
            TokenType.LBRACE, TokenType.RBRACE,
            TokenType.SEMICOLON, TokenType.COMMA,
        ]
        for tok, exp in zip(tokens, expected):
            assert tok.type == exp

    def test_line_tracking(self):
        src = "a\nb\nc"
        tokens = tokenize(src)
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3

    def test_skips_whitespace_and_comments(self):
        # If comments are supported, great; at minimum whitespace is skipped
        src = "  42  "
        tokens = tokenize(src)
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42

    def test_complex_expression(self):
        src = "let x = (1 + 2) * 3;"
        tokens = tokenize(src)
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.LET, TokenType.IDENT, TokenType.ASSIGN,
            TokenType.LPAREN, TokenType.NUMBER, TokenType.PLUS,
            TokenType.NUMBER, TokenType.RPAREN, TokenType.STAR,
            TokenType.NUMBER, TokenType.SEMICOLON,
        ]


# ============================================================
# Parser tests
# ============================================================


class TestParser:
    def _parse(self, src: str) -> Program:
        return parse(tokenize(src))

    def test_number_literal(self):
        prog = self._parse("42;")
        assert len(prog.stmts) == 1
        stmt = prog.stmts[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, NumberLit)
        assert stmt.expr.value == 42

    def test_string_literal(self):
        prog = self._parse('"hello";')
        stmt = prog.stmts[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, StringLit)
        assert stmt.expr.value == "hello"

    def test_bool_literals(self):
        prog = self._parse("true; false;")
        assert isinstance(prog.stmts[0], ExprStmt)
        assert isinstance(prog.stmts[0].expr, BoolLit)
        assert prog.stmts[0].expr.value is True
        assert isinstance(prog.stmts[1].expr, BoolLit)
        assert prog.stmts[1].expr.value is False

    def test_nil_literal(self):
        prog = self._parse("nil;")
        assert isinstance(prog.stmts[0].expr, NilLit)

    def test_binary_addition(self):
        prog = self._parse("1 + 2;")
        expr = prog.stmts[0].expr
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.left, NumberLit)
        assert isinstance(expr.right, NumberLit)

    def test_operator_precedence_mul_over_add(self):
        prog = self._parse("1 + 2 * 3;")
        expr = prog.stmts[0].expr
        # Should parse as 1 + (2 * 3)
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "*"

    def test_parenthesized_expression(self):
        prog = self._parse("(1 + 2) * 3;")
        expr = prog.stmts[0].expr
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == "+"

    def test_unary_negation(self):
        prog = self._parse("-5;")
        expr = prog.stmts[0].expr
        assert isinstance(expr, UnaryOp)
        assert expr.op == "-"

    def test_unary_not(self):
        prog = self._parse("not true;")
        expr = prog.stmts[0].expr
        assert isinstance(expr, UnaryOp)
        assert expr.op == "not"

    def test_let_statement(self):
        prog = self._parse("let x = 10;")
        stmt = prog.stmts[0]
        assert isinstance(stmt, LetStmt)
        assert stmt.name == "x"
        assert isinstance(stmt.value, NumberLit)

    def test_assignment_expression(self):
        prog = self._parse("x = 5;")
        stmt = prog.stmts[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, Assignment)
        assert stmt.expr.name == "x"

    def test_print_statement(self):
        prog = self._parse('print "hello";')
        stmt = prog.stmts[0]
        assert isinstance(stmt, PrintStmt)
        assert isinstance(stmt.value, StringLit)

    def test_if_statement(self):
        prog = self._parse("if true { 1; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, IfStmt)
        assert isinstance(stmt.condition, BoolLit)
        assert isinstance(stmt.then_block, Block)
        assert stmt.else_block is None

    def test_if_else_statement(self):
        prog = self._parse("if false { 1; } else { 2; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_block is not None

    def test_while_statement(self):
        prog = self._parse("while true { 1; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, WhileStmt)
        assert isinstance(stmt.condition, BoolLit)

    def test_function_declaration(self):
        prog = self._parse("fn add(a, b) { return a + b; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, FnDecl)
        assert stmt.name == "add"
        assert stmt.params == ["a", "b"]

    def test_function_no_params(self):
        prog = self._parse("fn greet() { print 1; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, FnDecl)
        assert stmt.params == []

    def test_return_statement_with_value(self):
        prog = self._parse("fn f() { return 42; }")
        fn = prog.stmts[0]
        ret = fn.body.stmts[0]
        assert isinstance(ret, ReturnStmt)
        assert isinstance(ret.value, NumberLit)

    def test_return_statement_no_value(self):
        prog = self._parse("fn f() { return; }")
        fn = prog.stmts[0]
        ret = fn.body.stmts[0]
        assert isinstance(ret, ReturnStmt)
        assert ret.value is None

    def test_call_expression(self):
        prog = self._parse("foo(1, 2);")
        expr = prog.stmts[0].expr
        assert isinstance(expr, CallExpr)
        assert isinstance(expr.callee, Identifier)
        assert expr.callee.name == "foo"
        assert len(expr.args) == 2

    def test_comparison_operators(self):
        for op in ["==", "!=", "<", ">", "<=", ">="]:
            prog = self._parse(f"1 {op} 2;")
            expr = prog.stmts[0].expr
            assert isinstance(expr, BinaryOp)
            assert expr.op == op

    def test_logical_and_or(self):
        prog = self._parse("true and false or true;")
        expr = prog.stmts[0].expr
        # or has lower precedence than and: (true and false) or true
        assert isinstance(expr, BinaryOp)
        assert expr.op == "or"

    def test_modulo(self):
        prog = self._parse("10 % 3;")
        expr = prog.stmts[0].expr
        assert isinstance(expr, BinaryOp)
        assert expr.op == "%"


# ============================================================
# Compiler tests
# ============================================================


class TestCompiler:
    def _compile(self, src: str) -> CompiledProgram:
        tokens = tokenize(src)
        ast = parse(tokens)
        return compile_ast(ast)

    def test_compile_number(self):
        prog = self._compile("42;")
        # Should have CONST instruction and HALT
        opcodes = [i.op for i in prog.code]
        assert OpCode.CONST in opcodes
        assert OpCode.HALT in opcodes

    def test_compile_addition(self):
        prog = self._compile("1 + 2;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.ADD in opcodes

    def test_compile_subtraction(self):
        prog = self._compile("5 - 3;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.SUB in opcodes

    def test_compile_multiplication(self):
        prog = self._compile("2 * 3;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.MUL in opcodes

    def test_compile_division(self):
        prog = self._compile("6 / 2;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.DIV in opcodes

    def test_compile_modulo(self):
        prog = self._compile("7 % 3;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.MOD in opcodes

    def test_compile_negation(self):
        prog = self._compile("-5;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.NEG in opcodes

    def test_compile_not(self):
        prog = self._compile("not true;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.NOT in opcodes

    def test_compile_comparison(self):
        prog = self._compile("1 == 2;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.EQ in opcodes

    def test_compile_let_and_load(self):
        prog = self._compile("let x = 10; x;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.STORE in opcodes
        assert OpCode.LOAD in opcodes

    def test_compile_print(self):
        prog = self._compile("print 42;")
        opcodes = [i.op for i in prog.code]
        assert OpCode.PRINT in opcodes

    def test_compile_if(self):
        prog = self._compile("if true { 1; }")
        opcodes = [i.op for i in prog.code]
        assert OpCode.JUMP_IF_FALSE in opcodes

    def test_compile_if_else(self):
        prog = self._compile("if false { 1; } else { 2; }")
        opcodes = [i.op for i in prog.code]
        assert OpCode.JUMP_IF_FALSE in opcodes
        assert OpCode.JUMP in opcodes

    def test_compile_while(self):
        prog = self._compile("while false { 1; }")
        opcodes = [i.op for i in prog.code]
        assert OpCode.JUMP_IF_FALSE in opcodes
        assert OpCode.JUMP in opcodes

    def test_compile_function(self):
        prog = self._compile("fn f() { return 1; }")
        assert "f" in prog.functions

    def test_compile_function_call(self):
        prog = self._compile("fn f() { return 1; } f();")
        opcodes = [i.op for i in prog.code]
        assert OpCode.CALL in opcodes

    def test_compile_return(self):
        prog = self._compile("fn f() { return 42; }")
        fn = prog.functions["f"]
        opcodes = [i.op for i in fn.code]
        assert OpCode.RETURN in opcodes


# ============================================================
# VM tests
# ============================================================


class TestVM:
    def _run(self, src: str) -> list[str]:
        tokens = tokenize(src)
        ast = parse(tokens)
        prog = compile_ast(ast)
        return run(prog)

    def test_print_number(self):
        result = self._run("print 42;")
        assert result == ["42"]

    def test_print_float(self):
        result = self._run("print 3.14;")
        assert result == ["3.14"]

    def test_print_string(self):
        result = self._run('print "hello";')
        assert result == ["hello"]

    def test_print_true(self):
        result = self._run("print true;")
        assert result == ["true"]

    def test_print_false(self):
        result = self._run("print false;")
        assert result == ["false"]

    def test_print_nil(self):
        result = self._run("print nil;")
        assert result == ["nil"]

    def test_addition(self):
        result = self._run("print 1 + 2;")
        assert result == ["3"]

    def test_subtraction(self):
        result = self._run("print 5 - 3;")
        assert result == ["2"]

    def test_multiplication(self):
        result = self._run("print 4 * 5;")
        assert result == ["20"]

    def test_division(self):
        result = self._run("print 10 / 3;")
        # integer division or float? spec says integer and float arithmetic
        # 10 / 3 with integers should yield float or truncated int
        # We'll accept the output as a string
        out = result[0]
        assert float(out) == pytest.approx(10 / 3, rel=1e-9)

    def test_modulo(self):
        result = self._run("print 10 % 3;")
        assert result == ["1"]

    def test_negation(self):
        result = self._run("print -5;")
        assert result == ["-5"]

    def test_not_true(self):
        result = self._run("print not true;")
        assert result == ["false"]

    def test_not_false(self):
        result = self._run("print not false;")
        assert result == ["true"]

    def test_operator_precedence(self):
        result = self._run("print 2 + 3 * 4;")
        assert result == ["14"]

    def test_parentheses(self):
        result = self._run("print (2 + 3) * 4;")
        assert result == ["20"]

    def test_string_concatenation(self):
        result = self._run('print "hello" + " " + "world";')
        assert result == ["hello world"]

    def test_comparison_eq(self):
        result = self._run("print 1 == 1;")
        assert result == ["true"]

    def test_comparison_ne(self):
        result = self._run("print 1 != 2;")
        assert result == ["true"]

    def test_comparison_lt(self):
        result = self._run("print 1 < 2;")
        assert result == ["true"]

    def test_comparison_gt(self):
        result = self._run("print 2 > 1;")
        assert result == ["true"]

    def test_comparison_le(self):
        result = self._run("print 2 <= 2;")
        assert result == ["true"]

    def test_comparison_ge(self):
        result = self._run("print 3 >= 2;")
        assert result == ["true"]

    def test_logical_and(self):
        result = self._run("print true and true;")
        assert result == ["true"]
        result = self._run("print true and false;")
        assert result == ["false"]

    def test_logical_or(self):
        result = self._run("print false or true;")
        assert result == ["true"]
        result = self._run("print false or false;")
        assert result == ["false"]

    def test_variable_binding(self):
        result = self._run("let x = 10; print x;")
        assert result == ["10"]

    def test_variable_reassignment(self):
        result = self._run("let x = 1; x = 2; print x;")
        assert result == ["2"]

    def test_multiple_variables(self):
        result = self._run("let a = 1; let b = 2; print a + b;")
        assert result == ["3"]

    def test_if_true(self):
        result = self._run("if true { print 1; }")
        assert result == ["1"]

    def test_if_false(self):
        result = self._run("if false { print 1; }")
        assert result == []

    def test_if_else_true(self):
        result = self._run("if true { print 1; } else { print 2; }")
        assert result == ["1"]

    def test_if_else_false(self):
        result = self._run("if false { print 1; } else { print 2; }")
        assert result == ["2"]

    def test_while_loop(self):
        src = """
        let i = 0;
        while i < 3 {
            print i;
            i = i + 1;
        }
        """
        result = self._run(src)
        assert result == ["0", "1", "2"]

    def test_nested_if(self):
        src = """
        let x = 10;
        if x > 5 {
            if x > 8 {
                print "big";
            }
        }
        """
        result = self._run(src)
        assert result == ["big"]

    def test_function_call_simple(self):
        src = """
        fn double(x) {
            return x * 2;
        }
        print double(5);
        """
        result = self._run(src)
        assert result == ["10"]

    def test_function_multiple_params(self):
        src = """
        fn add(a, b) {
            return a + b;
        }
        print add(3, 4);
        """
        result = self._run(src)
        assert result == ["7"]

    def test_function_no_params(self):
        src = """
        fn greet() {
            return "hi";
        }
        print greet();
        """
        result = self._run(src)
        assert result == ["hi"]

    def test_recursion_factorial(self):
        src = """
        fn fact(n) {
            if n <= 1 {
                return 1;
            }
            return n * fact(n - 1);
        }
        print fact(5);
        """
        result = self._run(src)
        assert result == ["120"]

    def test_recursion_fibonacci(self):
        src = """
        fn fib(n) {
            if n <= 1 {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }
        print fib(10);
        """
        result = self._run(src)
        assert result == ["55"]

    def test_function_reads_outer_variable(self):
        src = """
        let x = 100;
        fn getx() {
            return x;
        }
        print getx();
        """
        result = self._run(src)
        assert result == ["100"]

    def test_division_by_zero(self):
        with pytest.raises(VMError):
            self._run("print 1 / 0;")

    def test_modulo_by_zero(self):
        with pytest.raises(VMError):
            self._run("print 1 % 0;")

    def test_undefined_variable(self):
        with pytest.raises(VMError):
            self._run("print x;")

    def test_multiple_prints(self):
        src = """
        print 1;
        print 2;
        print 3;
        """
        result = self._run(src)
        assert result == ["1", "2", "3"]

    def test_complex_expression(self):
        result = self._run("print (10 - 2) * (3 + 1) / 4;")
        # (10-2)*(3+1)/4 = 8*4/4 = 8
        out = result[0]
        assert float(out) == pytest.approx(8.0)

    def test_nested_function_calls(self):
        src = """
        fn square(x) {
            return x * x;
        }
        fn sumsq(a, b) {
            return square(a) + square(b);
        }
        print sumsq(3, 4);
        """
        result = self._run(src)
        assert result == ["25"]

    def test_integer_output_no_trailing_dot_zero(self):
        """Integer results should print as integers (42 not 42.0)."""
        result = self._run("print 10 + 5;")
        assert result == ["15"]


# ============================================================
# Integration tests via execute()
# ============================================================


class TestExecute:
    def test_simple_expression(self):
        result = execute("print 1 + 2;")
        assert result == ["3"]

    def test_variable_and_print(self):
        result = execute("let x = 42; print x;")
        assert result == ["42"]

    def test_fizzbuzz(self):
        src = """
        let i = 1;
        while i <= 15 {
            if i % 15 == 0 {
                print "FizzBuzz";
            } else {
                if i % 3 == 0 {
                    print "Fizz";
                } else {
                    if i % 5 == 0 {
                        print "Buzz";
                    } else {
                        print i;
                    }
                }
            }
            i = i + 1;
        }
        """
        result = execute(src)
        expected = [
            "1", "2", "Fizz", "4", "Buzz",
            "Fizz", "7", "8", "Fizz", "Buzz",
            "11", "Fizz", "13", "14", "FizzBuzz",
        ]
        assert result == expected

    def test_bubble_sort(self):
        src = """
        let a = 5;
        let b = 3;
        let c = 8;
        let d = 1;
        let e = 4;

        let swapped = true;
        while swapped {
            swapped = false;
            if a > b { let t = a; a = b; b = t; swapped = true; }
            if b > c { let t = b; b = c; c = t; swapped = true; }
            if c > d { let t = c; c = d; d = t; swapped = true; }
            if d > e { let t = d; d = e; e = t; swapped = true; }
        }
        print a;
        print b;
        print c;
        print d;
        print e;
        """
        result = execute(src)
        assert result == ["1", "3", "4", "5", "8"]

    def test_gcd(self):
        src = """
        fn gcd(a, b) {
            while b != 0 {
                let t = b;
                b = a % b;
                a = t;
            }
            return a;
        }
        print gcd(48, 18);
        """
        result = execute(src)
        assert result == ["6"]

    def test_factorial_iterative(self):
        src = """
        fn factorial(n) {
            let result = 1;
            let i = 2;
            while i <= n {
                result = result * i;
                i = i + 1;
            }
            return result;
        }
        print factorial(6);
        """
        result = execute(src)
        assert result == ["720"]

    def test_string_building(self):
        src = """
        let s = "hello";
        let s2 = s + " world";
        print s2;
        """
        result = execute(src)
        assert result == ["hello world"]

    def test_boolean_logic_complex(self):
        src = """
        let a = true;
        let b = false;
        let c = true;
        print a and b or c;
        print (a or b) and (b or c);
        """
        result = execute(src)
        # a and b or c = (true and false) or true = false or true = true
        # (a or b) and (b or c) = true and true = true
        assert result == ["true", "true"]

    def test_nested_while(self):
        src = """
        let count = 0;
        let i = 0;
        while i < 3 {
            let j = 0;
            while j < 3 {
                count = count + 1;
                j = j + 1;
            }
            i = i + 1;
        }
        print count;
        """
        result = execute(src)
        assert result == ["9"]

    def test_function_with_local_vars(self):
        src = """
        fn calc(x) {
            let y = x + 1;
            let z = y * 2;
            return z;
        }
        print calc(4);
        """
        result = execute(src)
        assert result == ["10"]

    def test_multiple_function_calls(self):
        src = """
        fn inc(x) { return x + 1; }
        fn dec(x) { return x - 1; }
        print inc(5);
        print dec(5);
        print inc(dec(10));
        """
        result = execute(src)
        assert result == ["6", "4", "10"]

    def test_return_nil_implicitly(self):
        src = """
        fn noop() {
            let x = 1;
        }
        print noop();
        """
        result = execute(src)
        assert result == ["nil"]

    def test_comparison_chain(self):
        src = """
        print 1 < 2;
        print 2 > 1;
        print 3 >= 3;
        print 3 <= 3;
        print 1 == 1;
        print 1 != 2;
        """
        result = execute(src)
        assert result == ["true", "true", "true", "true", "true", "true"]
