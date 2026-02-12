"""Comprehensive tests for the bytecode compiler and VM."""

import pytest
from src.lang import execute
from src.lexer import tokenize, TokenType, LexerError
from src.parser import Parser, ParseError
from src.vm import RuntimeError_


# ---- Lexer Tests ----

class TestLexer:
    def test_numbers(self):
        tokens = tokenize("42 3.14")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "3.14"

    def test_strings(self):
        tokens = tokenize('"hello" "world"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
        assert tokens[1].type == TokenType.STRING
        assert tokens[1].value == "world"

    def test_keywords(self):
        tokens = tokenize("let if else while print fn return true false nil and or not")
        expected = [
            TokenType.LET, TokenType.IF, TokenType.ELSE, TokenType.WHILE,
            TokenType.PRINT, TokenType.FN, TokenType.RETURN, TokenType.TRUE,
            TokenType.FALSE, TokenType.NIL, TokenType.AND, TokenType.OR,
            TokenType.NOT, TokenType.EOF,
        ]
        for tok, exp in zip(tokens, expected):
            assert tok.type == exp

    def test_operators(self):
        tokens = tokenize("+ - * / % == != < > <= >= =")
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.EQ_EQ, TokenType.BANG_EQ, TokenType.LT,
            TokenType.GT, TokenType.LT_EQ, TokenType.GT_EQ, TokenType.EQ,
            TokenType.EOF,
        ]
        for tok, exp in zip(tokens, expected):
            assert tok.type == exp

    def test_delimiters(self):
        tokens = tokenize("( ) { } ; ,")
        expected = [
            TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE,
            TokenType.RBRACE, TokenType.SEMICOLON, TokenType.COMMA,
            TokenType.EOF,
        ]
        for tok, exp in zip(tokens, expected):
            assert tok.type == exp

    def test_identifiers(self):
        tokens = tokenize("foo bar_baz _test x1")
        for i in range(4):
            assert tokens[i].type == TokenType.IDENT

    def test_line_tracking(self):
        tokens = tokenize("a\nb\nc")
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3

    def test_unterminated_string(self):
        with pytest.raises(LexerError):
            tokenize('"hello')

    def test_comments(self):
        tokens = tokenize("a // comment\nb")
        assert len(tokens) == 3  # a, b, EOF
        assert tokens[0].value == "a"
        assert tokens[1].value == "b"


# ---- Arithmetic & Precedence ----

class TestArithmetic:
    def test_basic_add(self):
        assert execute("print 1 + 2;") == ["3"]

    def test_basic_sub(self):
        assert execute("print 5 - 3;") == ["2"]

    def test_basic_mul(self):
        assert execute("print 3 * 4;") == ["12"]

    def test_basic_div(self):
        assert execute("print 10 / 3;") == ["3"]

    def test_float_div(self):
        assert execute("print 7.0 / 2.0;") == ["3.5"]

    def test_modulo(self):
        assert execute("print 10 % 3;") == ["1"]

    def test_precedence(self):
        assert execute("print 2 + 3 * 4;") == ["14"]

    def test_parens(self):
        assert execute("print (2 + 3) * 4;") == ["20"]

    def test_unary_neg(self):
        assert execute("print -5;") == ["-5"]

    def test_nested_arithmetic(self):
        assert execute("print (10 - 2) * (3 + 1) / 4;") == ["8"]

    def test_multiple_prints(self):
        result = execute("print 1; print 2; print 3;")
        assert result == ["1", "2", "3"]


# ---- Variables ----

class TestVariables:
    def test_let_binding(self):
        assert execute("let x = 10; print x;") == ["10"]

    def test_variable_update(self):
        assert execute("let x = 1; x = 2; print x;") == ["2"]

    def test_multiple_vars(self):
        code = "let a = 1; let b = 2; print a + b;"
        assert execute(code) == ["3"]

    def test_var_in_expression(self):
        code = "let x = 5; let y = x * 2; print y;"
        assert execute(code) == ["10"]

    def test_undefined_variable(self):
        with pytest.raises(RuntimeError_):
            execute("print x;")


# ---- Control Flow ----

class TestControlFlow:
    def test_if_true(self):
        assert execute("if true { print 1; }") == ["1"]

    def test_if_false(self):
        assert execute("if false { print 1; }") == []

    def test_if_else(self):
        assert execute("if false { print 1; } else { print 2; }") == ["2"]

    def test_if_comparison(self):
        assert execute("let x = 5; if x > 3 { print 1; } else { print 0; }") == ["1"]

    def test_while_loop(self):
        code = """
        let i = 0;
        while i < 5 {
            print i;
            i = i + 1;
        }
        """
        assert execute(code) == ["0", "1", "2", "3", "4"]

    def test_nested_if(self):
        code = """
        let x = 10;
        if x > 5 {
            if x > 8 {
                print 1;
            } else {
                print 0;
            }
        }
        """
        assert execute(code) == ["1"]

    def test_while_with_break_condition(self):
        code = """
        let sum = 0;
        let i = 1;
        while i <= 10 {
            sum = sum + i;
            i = i + 1;
        }
        print sum;
        """
        assert execute(code) == ["55"]


# ---- Boolean Logic ----

class TestBooleans:
    def test_and_true(self):
        assert execute("print true and true;") == ["true"]

    def test_and_false(self):
        assert execute("print true and false;") == ["false"]

    def test_or_true(self):
        assert execute("print false or true;") == ["true"]

    def test_or_false(self):
        assert execute("print false or false;") == ["false"]

    def test_not_true(self):
        assert execute("print not true;") == ["false"]

    def test_not_false(self):
        assert execute("print not false;") == ["true"]

    def test_comparison_eq(self):
        assert execute("print 1 == 1;") == ["true"]

    def test_comparison_ne(self):
        assert execute("print 1 != 2;") == ["true"]

    def test_comparison_lt(self):
        assert execute("print 1 < 2;") == ["true"]

    def test_nil_is_falsy(self):
        assert execute("if not nil { print 1; }") == ["1"]


# ---- Strings ----

class TestStrings:
    def test_string_print(self):
        assert execute('print "hello";') == ["hello"]

    def test_string_concat(self):
        assert execute('print "hello" + " " + "world";') == ["hello world"]

    def test_string_comparison(self):
        assert execute('print "abc" == "abc";') == ["true"]


# ---- Functions ----

class TestFunctions:
    def test_basic_function(self):
        code = """
        fn add(a, b) {
            return a + b;
        }
        print add(3, 4);
        """
        assert execute(code) == ["7"]

    def test_no_args_function(self):
        code = """
        fn greet() {
            return 42;
        }
        print greet();
        """
        assert execute(code) == ["42"]

    def test_recursive_factorial(self):
        code = """
        fn factorial(n) {
            if n <= 1 {
                return 1;
            }
            return n * factorial(n - 1);
        }
        print factorial(5);
        """
        assert execute(code) == ["120"]

    def test_recursive_fibonacci(self):
        code = """
        fn fib(n) {
            if n <= 1 {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }
        print fib(10);
        """
        assert execute(code) == ["55"]

    def test_multiple_functions(self):
        code = """
        fn double(x) { return x * 2; }
        fn triple(x) { return x * 3; }
        print double(5);
        print triple(5);
        """
        assert execute(code) == ["10", "15"]

    def test_function_with_local_vars(self):
        code = """
        fn compute(x) {
            let y = x * 2;
            let z = y + 1;
            return z;
        }
        print compute(5);
        """
        assert execute(code) == ["11"]


# ---- Error Cases ----

class TestErrors:
    def test_division_by_zero(self):
        with pytest.raises(RuntimeError_):
            execute("print 1 / 0;")

    def test_mod_by_zero(self):
        with pytest.raises(RuntimeError_):
            execute("print 1 % 0;")

    def test_type_error_add(self):
        with pytest.raises(RuntimeError_):
            execute("print true + 1;")

    def test_undefined_var(self):
        with pytest.raises(RuntimeError_):
            execute("print undefined_var;")

    def test_call_non_function(self):
        with pytest.raises(RuntimeError_):
            execute("let x = 5; x();")

    def test_wrong_arg_count(self):
        with pytest.raises(RuntimeError_):
            execute("fn f(a) { return a; } f(1, 2);")


# ---- Complex Programs ----

class TestComplexPrograms:
    def test_fizzbuzz(self):
        code = """
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
        expected = [
            "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
            "11", "Fizz", "13", "14", "FizzBuzz"
        ]
        assert execute(code) == expected

    def test_gcd(self):
        code = """
        fn gcd(a, b) {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }
        print gcd(48, 18);
        """
        assert execute(code) == ["6"]

    def test_sum_function(self):
        code = """
        fn sum_to(n) {
            let total = 0;
            let i = 1;
            while i <= n {
                total = total + i;
                i = i + 1;
            }
            return total;
        }
        print sum_to(100);
        """
        assert execute(code) == ["5050"]

    def test_nested_function_calls(self):
        code = """
        fn square(x) { return x * x; }
        fn sum_squares(a, b) { return square(a) + square(b); }
        print sum_squares(3, 4);
        """
        assert execute(code) == ["25"]

    def test_counter(self):
        code = """
        let count = 0;
        fn increment() {
            count = count + 1;
            return count;
        }
        print increment();
        print increment();
        print increment();
        """
        assert execute(code) == ["1", "2", "3"]

    def test_power_function(self):
        code = """
        fn power(base, exp) {
            if exp == 0 { return 1; }
            return base * power(base, exp - 1);
        }
        print power(2, 10);
        """
        assert execute(code) == ["1024"]
