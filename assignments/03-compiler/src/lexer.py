from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


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
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMI = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int


class LexerError(Exception):
    pass


class Lexer:
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

    SINGLE_CHAR = {
        "+": TokenType.PLUS,
        "-": TokenType.MINUS,
        "*": TokenType.STAR,
        "/": TokenType.SLASH,
        "%": TokenType.PERCENT,
        "(": TokenType.LPAREN,
        ")": TokenType.RPAREN,
        "{": TokenType.LBRACE,
        "}": TokenType.RBRACE,
        ",": TokenType.COMMA,
        ";": TokenType.SEMI,
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1

    def _peek(self) -> str | None:
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
        return ch

    def _read_number(self) -> Token:
        start = self.pos
        line = self.line
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.pos += 1
        if self.pos < len(self.source) and self.source[self.pos] == ".":
            self.pos += 1  # consume '.'
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
            return Token(TokenType.NUMBER, float(self.source[start : self.pos]), line)
        return Token(TokenType.NUMBER, int(self.source[start : self.pos]), line)

    def _read_string(self) -> Token:
        line = self.line
        self._advance()  # consume opening '"'
        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos] != '"':
            if self.source[self.pos] == "\n":
                self.line += 1
            self.pos += 1
        if self.pos >= len(self.source):
            raise LexerError(f"Unterminated string at line {line}")
        value = self.source[start : self.pos]
        self.pos += 1  # consume closing '"'
        return Token(TokenType.STRING, value, line)

    def _read_ident(self) -> Token:
        start = self.pos
        line = self.line
        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or self.source[self.pos] == "_"
        ):
            self.pos += 1
        word = self.source[start : self.pos]
        token_type = self.KEYWORDS.get(word, TokenType.IDENT)
        return Token(token_type, word, line)

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while self.pos < len(self.source):
            ch = self.source[self.pos]

            # Skip whitespace, track newlines
            if ch in (" ", "\t", "\r", "\n"):
                self._advance()
                continue

            # Numbers
            if ch.isdigit():
                tokens.append(self._read_number())
                continue

            # Strings
            if ch == '"':
                tokens.append(self._read_string())
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_ident())
                continue

            # Two-char operators
            line = self.line
            if ch == "=" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                tokens.append(Token(TokenType.EQ, "==", line))
                self.pos += 2
                continue
            if ch == "!" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                tokens.append(Token(TokenType.NE, "!=", line))
                self.pos += 2
                continue
            if ch == "<" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                tokens.append(Token(TokenType.LE, "<=", line))
                self.pos += 2
                continue
            if ch == ">" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                tokens.append(Token(TokenType.GE, ">=", line))
                self.pos += 2
                continue

            # Single-char: = < >
            if ch == "=":
                tokens.append(Token(TokenType.ASSIGN, "=", line))
                self.pos += 1
                continue
            if ch == "<":
                tokens.append(Token(TokenType.LT, "<", line))
                self.pos += 1
                continue
            if ch == ">":
                tokens.append(Token(TokenType.GT, ">", line))
                self.pos += 1
                continue

            # Other single-char ops and delimiters
            if ch in self.SINGLE_CHAR:
                tokens.append(Token(self.SINGLE_CHAR[ch], ch, line))
                self.pos += 1
                continue

            raise LexerError(f"Unexpected character '{ch}' at line {self.line}")

        tokens.append(Token(TokenType.EOF, None, self.line))
        return tokens
