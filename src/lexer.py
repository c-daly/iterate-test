"""Lexer for the toy language.

Produces a flat list of Token objects from source text.
Each Token has type (str), value (any), line (int), col (int).
"""
from dataclasses import dataclass
from typing import Any, List

NEWLINE = chr(10)
TAB = chr(9)
CR = chr(13)
QUOTE = chr(34)
BACKSLASH = chr(92)

KEYWORDS = {
    "let", "if", "else", "while", "print", "fn", "return",
    "true", "false", "nil", "and", "or", "not",
}

MULTI_CHAR_OPS = ["==", "!=", "<=", ">="]
SINGLE_CHAR_OPS = set("+-*/%<>=!")
PUNCT = set("(){},;")


@dataclass
class Token:
    type: str
    value: Any
    line: int
    col: int


class LexError(Exception):
    pass


class Lexer:
    def __init__(self, source: str) -> None:
        self.src = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    def _peek(self, off: int = 0) -> str:
        p = self.pos + off
        return self.src[p] if p < len(self.src) else ""

    def _advance(self) -> str:
        ch = self.src[self.pos]
        self.pos += 1
        if ch == NEWLINE:
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _emit(self, type_: str, value: Any, line: int, col: int) -> None:
        self.tokens.append(Token(type_, value, line, col))

    def tokenize(self) -> List[Token]:
        whitespace = " " + TAB + CR + NEWLINE
        while self.pos < len(self.src):
            ch = self._peek()
            if ch in whitespace:
                self._advance()
                continue
            if ch == "/" and self._peek(1) == "/":
                while self.pos < len(self.src) and self._peek() != NEWLINE:
                    self._advance()
                continue
            line, col = self.line, self.col
            if ch.isdigit():
                self._number(line, col)
            elif ch == QUOTE:
                self._string(line, col)
            elif ch.isalpha() or ch == "_":
                self._ident(line, col)
            elif ch in PUNCT:
                self._advance()
                self._emit(ch, ch, line, col)
            elif ch in SINGLE_CHAR_OPS:
                two = self.src[self.pos:self.pos + 2]
                if two in MULTI_CHAR_OPS:
                    self._advance()
                    self._advance()
                    self._emit(two, two, line, col)
                else:
                    self._advance()
                    self._emit(ch, ch, line, col)
            else:
                raise LexError("Unexpected char at line " + str(line) + " col " + str(col))
        self._emit("EOF", None, self.line, self.col)
        return self.tokens

    def _number(self, line: int, col: int) -> None:
        start = self.pos
        while self._peek().isdigit():
            self._advance()
        is_float = False
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while self._peek().isdigit():
                self._advance()
        text = self.src[start:self.pos]
        value: Any = float(text) if is_float else int(text)
        self._emit("NUMBER", value, line, col)

    def _string(self, line: int, col: int) -> None:
        self._advance()
        chars: List[str] = []
        escapes = {"n": NEWLINE, "t": TAB, "r": CR, BACKSLASH: BACKSLASH, QUOTE: QUOTE}
        while self.pos < len(self.src) and self._peek() != QUOTE:
            ch = self._advance()
            if ch == BACKSLASH:
                nxt = self._advance() if self.pos < len(self.src) else ""
                chars.append(escapes.get(nxt, nxt))
            else:
                chars.append(ch)
        if self.pos >= len(self.src):
            raise LexError("Unterminated string at line " + str(line) + " col " + str(col))
        self._advance()
        self._emit("STRING", "".join(chars), line, col)

    def _ident(self, line: int, col: int) -> None:
        start = self.pos
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        text = self.src[start:self.pos]
        if text in KEYWORDS:
            self._emit(text, text, line, col)
        else:
            self._emit("IDENT", text, line, col)


def tokenize(source: str) -> List[Token]:
    return Lexer(source).tokenize()
