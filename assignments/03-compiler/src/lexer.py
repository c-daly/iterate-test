"""Lexer: tokenizes source code into a token stream."""

from enum import Enum, auto
from typing import NamedTuple


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()

    # Keywords
    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    PRINT = auto()
    FN = auto()
    RETURN = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    ASSIGN = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()

    # Special
    EOF = auto()


class Token(NamedTuple):
    type: TokenType
    value: object
    line: int


KEYWORDS = {
    "let": TokenType.LET,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "print": TokenType.PRINT,
    "fn": TokenType.FN,
    "return": TokenType.RETURN,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "nil": TokenType.NIL,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
}


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    n = len(source)

    while i < n:
        c = source[i]

        # Whitespace
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r":
            i += 1
            continue

        # Comments (// to end of line)
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # Numbers
        if c.isdigit():
            start = i
            while i < n and source[i].isdigit():
                i += 1
            if i < n and source[i] == "." and i + 1 < n and source[i + 1].isdigit():
                i += 1  # skip dot
                while i < n and source[i].isdigit():
                    i += 1
                tokens.append(Token(TokenType.NUMBER, float(source[start:i]), line))
            else:
                tokens.append(Token(TokenType.NUMBER, int(source[start:i]), line))
            continue

        # Strings
        if c == '"':
            i += 1
            start = i
            while i < n and source[i] != '"':
                if source[i] == "\n":
                    line += 1
                i += 1
            tokens.append(Token(TokenType.STRING, source[start:i], line))
            i += 1  # skip closing quote
            continue

        # Identifiers and keywords
        if c.isalpha() or c == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]
            tt = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line))
            continue

        # Two-character operators
        if i + 1 < n:
            two = source[i : i + 2]
            if two == "==":
                tokens.append(Token(TokenType.EQ, "==", line))
                i += 2
                continue
            if two == "!=":
                tokens.append(Token(TokenType.NE, "!=", line))
                i += 2
                continue
            if two == "<=":
                tokens.append(Token(TokenType.LE, "<=", line))
                i += 2
                continue
            if two == ">=":
                tokens.append(Token(TokenType.GE, ">=", line))
                i += 2
                continue

        # Single-character tokens
        single = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "=": TokenType.ASSIGN,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            ";": TokenType.SEMICOLON,
            ",": TokenType.COMMA,
        }
        if c in single:
            tokens.append(Token(single[c], c, line))
            i += 1
            continue

        raise SyntaxError(f"Unexpected character {c!r} at line {line}")

    tokens.append(Token(TokenType.EOF, None, line))
    return tokens
