"""Lexer - tokenizes source code into a token stream."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


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
    EQ_EQ = auto()
    BANG_EQ = auto()
    LT = auto()
    GT = auto()
    LT_EQ = auto()
    GT_EQ = auto()
    EQ = auto()
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()
    # Special
    EOF = auto()


KEYWORDS = {
    "let": TokenType.LET, "if": TokenType.IF, "else": TokenType.ELSE,
    "while": TokenType.WHILE, "print": TokenType.PRINT, "fn": TokenType.FN,
    "return": TokenType.RETURN, "true": TokenType.TRUE, "false": TokenType.FALSE,
    "nil": TokenType.NIL, "and": TokenType.AND, "or": TokenType.OR, "not": TokenType.NOT,
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"Lexer error at {line}:{col}: {message}")
        self.line = line
        self.col = col


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    line = 1
    col = 1
    n = len(source)

    while pos < n:
        ch = source[pos]

        if ch in " \t\r":
            pos += 1
            col += 1
            continue
        if ch == "\n":
            pos += 1
            line += 1
            col = 1
            continue

        # Line comments
        if ch == "/" and pos + 1 < n and source[pos + 1] == "/":
            pos += 2
            while pos < n and source[pos] != "\n":
                pos += 1
            continue

        start_col = col

        # Numbers
        if ch.isdigit():
            start = pos
            while pos < n and source[pos].isdigit():
                pos += 1
                col += 1
            if pos < n and source[pos] == "." and pos + 1 < n and source[pos + 1].isdigit():
                pos += 1
                col += 1
                while pos < n and source[pos].isdigit():
                    pos += 1
                    col += 1
            tokens.append(Token(TokenType.NUMBER, source[start:pos], line, start_col))
            continue

        # Strings
        if ch == '"':
            start = pos
            pos += 1
            col += 1
            while pos < n and source[pos] != '"':
                if source[pos] == "\n":
                    raise LexerError("Unterminated string", line, start_col)
                if source[pos] == "\\" and pos + 1 < n:
                    pos += 1
                    col += 1
                pos += 1
                col += 1
            if pos >= n:
                raise LexerError("Unterminated string", line, start_col)
            pos += 1
            col += 1
            raw = source[start + 1 : pos - 1]
            # Process escape sequences
            value = raw.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")
            tokens.append(Token(TokenType.STRING, value, line, start_col))
            continue

        # Identifiers / keywords
        if ch.isalpha() or ch == "_":
            start = pos
            while pos < n and (source[pos].isalnum() or source[pos] == "_"):
                pos += 1
                col += 1
            word = source[start:pos]
            tt = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line, start_col))
            continue

        # Two-char operators
        if pos + 1 < n:
            two = source[pos : pos + 2]
            two_map = {"==": TokenType.EQ_EQ, "!=": TokenType.BANG_EQ, "<=": TokenType.LT_EQ, ">=": TokenType.GT_EQ}
            if two in two_map:
                tokens.append(Token(two_map[two], two, line, start_col))
                pos += 2
                col += 2
                continue

        # Single-char tokens
        one_map = {
            "+": TokenType.PLUS, "-": TokenType.MINUS, "*": TokenType.STAR,
            "/": TokenType.SLASH, "%": TokenType.PERCENT, "<": TokenType.LT,
            ">": TokenType.GT, "=": TokenType.EQ, "(": TokenType.LPAREN,
            ")": TokenType.RPAREN, "{": TokenType.LBRACE, "}": TokenType.RBRACE,
            ";": TokenType.SEMICOLON, ",": TokenType.COMMA,
        }
        if ch in one_map:
            tokens.append(Token(one_map[ch], ch, line, start_col))
            pos += 1
            col += 1
            continue

        raise LexerError(f"Unexpected character: {ch!r}", line, start_col)

    tokens.append(Token(TokenType.EOF, "", line, col))
    return tokens
