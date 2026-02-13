import pytest
from lexer import Lexer, Token, TokenType, LexerError
from parser import (
    Parser, ParseError,
    NumberLit, StringLit, BoolLit, NilLit, Identifier,
    UnaryOp, BinOp, Assignment, LetStmt, IfStmt, WhileStmt,
    PrintStmt, FnDecl, ReturnStmt, CallExpr, ExprStmt,
)


class TestLexNumbers:
    def test_lex_numbers(self):
        tokens = Lexer("42 3.14").tokenize()
        assert tokens[0] == Token(TokenType.NUMBER, 42, 1)
        assert tokens[1] == Token(TokenType.NUMBER, 3.14, 1)
        assert tokens[2].type == TokenType.EOF


class TestLexStrings:
    def test_lex_strings(self):
        tokens = Lexer('"hello"').tokenize()
        assert tokens[0] == Token(TokenType.STRING, "hello", 1)
        assert tokens[1].type == TokenType.EOF


class TestLexKeywords:
    EXPECTED = {
        "let": TokenType.LET,
        "if": TokenType.IF,
        "else": TokenType.ELSE,
        "while": TokenType.WHILE,
        "fn": TokenType.FN,
        "return": TokenType.RETURN,
        "print": TokenType.PRINT,
        "true": TokenType.TRUE,
        "false": TokenType.FALSE,
        "nil": TokenType.NIL,
        "and": TokenType.AND,
        "or": TokenType.OR,
        "not": TokenType.NOT,
    }

    def test_lex_keywords(self):
        source = " ".join(self.EXPECTED.keys())
        tokens = Lexer(source).tokenize()
        for i, (kw, expected_type) in enumerate(self.EXPECTED.items()):
            assert tokens[i].type == expected_type, f"keyword '{kw}' got {tokens[i].type}"
        assert tokens[len(self.EXPECTED)].type == TokenType.EOF


class TestLexOperators:
    def test_lex_operators(self):
        source = "+ - * / % == != < > <= >= ="
        tokens = Lexer(source).tokenize()
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.EQ, TokenType.NE, TokenType.LT,
            TokenType.GT, TokenType.LE, TokenType.GE, TokenType.ASSIGN,
        ]
        for i, et in enumerate(expected):
            assert tokens[i].type == et, f"index {i}: expected {et}, got {tokens[i].type}"
        assert tokens[len(expected)].type == TokenType.EOF


class TestLexDelimiters:
    def test_lex_delimiters(self):
        source = "( ) { } , ;"
        tokens = Lexer(source).tokenize()
        expected = [
            TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE,
            TokenType.RBRACE, TokenType.COMMA, TokenType.SEMI,
        ]
        for i, et in enumerate(expected):
            assert tokens[i].type == et, f"index {i}: expected {et}, got {tokens[i].type}"
        assert tokens[len(expected)].type == TokenType.EOF


class TestLexExpression:
    def test_lex_expression(self):
        tokens = Lexer("let x = 42;").tokenize()
        assert tokens[0].type == TokenType.LET
        assert tokens[1] == Token(TokenType.IDENT, "x", 1)
        assert tokens[2].type == TokenType.ASSIGN
        assert tokens[3] == Token(TokenType.NUMBER, 42, 1)
        assert tokens[4].type == TokenType.SEMI
        assert tokens[5].type == TokenType.EOF


class TestLexMultiline:
    def test_lex_multiline(self):
        source = "let x = 1;\nlet y = 2;\nlet z = 3;"
        tokens = Lexer(source).tokenize()
        # First line tokens should be line 1
        assert tokens[0].line == 1  # let
        # After first newline, line 2
        assert tokens[5].line == 2  # let (second)
        # After second newline, line 3
        assert tokens[10].line == 3  # let (third)


class TestLexError:
    def test_lex_error_unterminated_string(self):
        with pytest.raises(LexerError):
            Lexer('"unterminated').tokenize()


# --- Helper ---

def _parse(source: str) -> list:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# --- Parser Tests ---

class TestParseLet:
    def test_parse_let(self):
        stmts = _parse("let x = 42;")
        assert len(stmts) == 1
        assert stmts[0] == LetStmt("x", NumberLit(42))


class TestParseArithmetic:
    def test_parse_arithmetic(self):
        stmts = _parse("1 + 2 * 3;")
        assert len(stmts) == 1
        expr = stmts[0].expr
        # 1 + (2 * 3) due to precedence
        assert isinstance(expr, BinOp)
        assert expr.op == "+"
        assert expr.left == NumberLit(1)
        assert isinstance(expr.right, BinOp)
        assert expr.right.op == "*"
        assert expr.right.left == NumberLit(2)
        assert expr.right.right == NumberLit(3)


class TestParseIfElse:
    def test_parse_if_else(self):
        stmts = _parse("if x { print 1; } else { print 2; }")
        assert len(stmts) == 1
        stmt = stmts[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.condition == Identifier("x")
        assert len(stmt.then_body) == 1
        assert isinstance(stmt.then_body[0], PrintStmt)
        assert stmt.then_body[0].value == NumberLit(1)
        assert stmt.else_body is not None
        assert len(stmt.else_body) == 1
        assert isinstance(stmt.else_body[0], PrintStmt)
        assert stmt.else_body[0].value == NumberLit(2)


class TestParseWhile:
    def test_parse_while(self):
        stmts = _parse("while x { print x; }")
        assert len(stmts) == 1
        stmt = stmts[0]
        assert isinstance(stmt, WhileStmt)
        assert stmt.condition == Identifier("x")
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], PrintStmt)


class TestParseFunction:
    def test_parse_function(self):
        stmts = _parse("fn add(a, b) { return a + b; }")
        assert len(stmts) == 1
        fn = stmts[0]
        assert isinstance(fn, FnDecl)
        assert fn.name == "add"
        assert fn.params == ["a", "b"]
        assert len(fn.body) == 1
        ret = fn.body[0]
        assert isinstance(ret, ReturnStmt)
        assert isinstance(ret.value, BinOp)
        assert ret.value.op == "+"
        assert ret.value.left == Identifier("a")
        assert ret.value.right == Identifier("b")


class TestParseCall:
    def test_parse_call(self):
        stmts = _parse("add(1, 2);")
        assert len(stmts) == 1
        stmt = stmts[0]
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, CallExpr)
        assert call.callee == Identifier("add")
        assert call.args == [NumberLit(1), NumberLit(2)]


class TestParseNested:
    def test_parse_nested(self):
        source = "if true { if false { print 1; } }"
        stmts = _parse(source)
        assert len(stmts) == 1
        outer = stmts[0]
        assert isinstance(outer, IfStmt)
        assert outer.condition == BoolLit(True)
        assert len(outer.then_body) == 1
        inner = outer.then_body[0]
        assert isinstance(inner, IfStmt)
        assert inner.condition == BoolLit(False)
        assert len(inner.then_body) == 1
        assert isinstance(inner.then_body[0], PrintStmt)
        assert inner.else_body is None


class TestParseError:
    def test_parse_error(self):
        with pytest.raises(ParseError):
            _parse("let x = 42")
