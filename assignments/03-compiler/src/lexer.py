"""Lexer for the toy language.

Produces a list of Token objects from a source string.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List

NL = chr(10)
TAB = chr(9)
CR = chr(13)
BSLASH = chr(92)
DQUOTE = chr(34)
WHITESPACE = " " + TAB + CR + NL


class TokenType(Enum):
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FN = auto()
    RETURN = auto()
    PRINT = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMI = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int


KEYWORDS = {
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

_TWO_CHAR_OPS = [
    ("=", "=", TokenType.ASSIGN, TokenType.EQ),
    ("!", "=", None, TokenType.NE),
    ("<", "=", TokenType.LT, TokenType.LE),
    (">", "=", TokenType.GT, TokenType.GE),
]

_SINGLE_CHAR_OPS = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ",": TokenType.COMMA,
    ";": TokenType.SEMI,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
}

_STRING_ESCAPES = {"n": NL, "t": TAB, "r": CR, BSLASH: BSLASH, DQUOTE: DQUOTE}


class LexError(Exception):
    """Raised on a malformed source token."""


class Lexer:
    """Converts source text into a list of Tokens."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.pos = 0
        self.line = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        while not self._at_end():
            self._skip_trivia()
            if self._at_end():
                break
            ch = self._peek()
            if ch.isdigit():
                self._number()
            elif ch == DQUOTE:
                self._string()
            elif ch.isalpha() or ch == "_":
                self._identifier()
            else:
                self._operator()
        self.tokens.append(Token(TokenType.EOF, None, self.line))
        return self.tokens

    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx >= len(self.source):
            return ""
        return self.source[idx]

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == NL:
            self.line += 1
        return ch

    def _skip_trivia(self) -> None:
        while not self._at_end():
            ch = self._peek()
            if ch in WHITESPACE:
                self._advance()
            elif ch == "/" and self._peek(1) == "/":
                while not self._at_end() and self._peek() != NL:
                    self._advance()
            else:
                break

    def _number(self) -> None:
        start_line = self.line
        start = self.pos
        while not self._at_end() and self._peek().isdigit():
            self._advance()
        is_float = False
        if not self._at_end() and self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while not self._at_end() and self._peek().isdigit():
                self._advance()
        text = self.source[start:self.pos]
        value: Any = float(text) if is_float else int(text)
        self.tokens.append(Token(TokenType.NUMBER, value, start_line))

    def _string(self) -> None:
        start_line = self.line
        self._advance()
        buf: List[str] = []
        while not self._at_end() and self._peek() != DQUOTE:
            ch = self._advance()
            if ch == BSLASH and not self._at_end():
                esc = self._advance()
                buf.append(_STRING_ESCAPES.get(esc, esc))
            else:
                buf.append(ch)
        if self._at_end():
            raise LexError(f"Unterminated string starting at line {start_line}")
        self._advance()
        self.tokens.append(Token(TokenType.STRING, "".join(buf), start_line))

    def _identifier(self) -> None:
        start_line = self.line
        start = self.pos
        while not self._at_end() and (self._peek().isalnum() or self._peek() == "_"):
            self._advance()
        text = self.source[start:self.pos]
        kind = KEYWORDS.get(text, TokenType.IDENT)
        value = text if kind == TokenType.IDENT else None
        self.tokens.append(Token(kind, value, start_line))

    def _operator(self) -> None:
        start_line = self.line
        ch = self._peek()
        nxt = self._peek(1)
        for first, second, single_type, double_type in _TWO_CHAR_OPS:
            if ch == first:
                if nxt == second:
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(double_type, None, start_line))
                    return
                if single_type is not None:
                    self._advance()
                    self.tokens.append(Token(single_type, None, start_line))
                    return
                raise LexError(f"Unexpected character {ch!r} at line {start_line}")
        if ch in _SINGLE_CHAR_OPS:
            self._advance()
            self.tokens.append(Token(_SINGLE_CHAR_OPS[ch], None, start_line))
            return
        raise LexError(f"Unexpected character {ch!r} at line {start_line}")
