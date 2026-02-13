import pytest
from lexer import Lexer, Token, TokenType, LexerError


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
