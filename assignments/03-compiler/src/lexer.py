"""Lexer: turn source text into a flat list of Token objects.

The token stream is consumed by parser.py. The grammar (see the assignment
README) needs identifiers, number literals (int and float), string literals,
keywords, and a fixed set of operators / punctuation.
"""


class LexError(Exception):
    """Raised when the source contains a character we cannot tokenize."""


class Token:
    __slots__ = ("type", "value", "line")

    def __init__(self, type_, value, line):
        self.type = type_
        self.value = value
        self.line = line

    def __repr__(self):
        return "Token(%r, %r, line=%d)" % (self.type, self.value, self.line)

    def __eq__(self, other):
        return (
            isinstance(other, Token)
            and self.type == other.type
            and self.value == other.value
        )


# Reserved words map to their own token type so the parser can switch on them.
KEYWORDS = {
    "let": "LET",
    "if": "IF",
    "else": "ELSE",
    "while": "WHILE",
    "print": "PRINT",
    "fn": "FN",
    "return": "RETURN",
    "true": "TRUE",
    "false": "FALSE",
    "nil": "NIL",
    "and": "AND",
    "or": "OR",
    "not": "NOT",
}

# Two-character operators must be checked before single-character ones.
TWO_CHAR = {
    "==": "EQ",
    "!=": "NE",
    "<=": "LE",
    ">=": "GE",
}

ONE_CHAR = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",
    "/": "SLASH",
    "%": "PERCENT",
    "<": "LT",
    ">": "GT",
    "=": "ASSIGN",
    "(": "LPAREN",
    ")": "RPAREN",
    "{": "LBRACE",
    "}": "RBRACE",
    ",": "COMMA",
    ";": "SEMI",
}


def tokenize(source):
    """Return a list of Tokens ending in a single EOF token."""
    tokens = []
    i = 0
    line = 1
    n = len(source)

    while i < n:
        c = source[i]

        # Whitespace -----------------------------------------------------
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r":
            i += 1
            continue

        # Line comments: // ... to end of line ---------------------------
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # String literals -----------------------------------------------
        if c == "\"":
            i += 1
            start_line = line
            chars = []
            while i < n and source[i] != "\"":
                ch = source[i]
                if ch == "\\" and i + 1 < n:
                    nxt = source[i + 1]
                    chars.append(
                        {"n": "\n", "t": "\t", "\"": "\"", "\\": "\\"}.get(nxt, nxt)
                    )
                    i += 2
                    continue
                if ch == "\n":
                    line += 1
                chars.append(ch)
                i += 1
            if i >= n:
                raise LexError("Unterminated string starting on line %d" % start_line)
            i += 1  # closing quote
            tokens.append(Token("STRING", "".join(chars), start_line))
            continue

        # Numbers (int or float) ----------------------------------------
        if c.isdigit():
            start = i
            while i < n and source[i].isdigit():
                i += 1
            is_float = False
            if i < n and source[i] == "." and i + 1 < n and source[i + 1].isdigit():
                is_float = True
                i += 1
                while i < n and source[i].isdigit():
                    i += 1
            text = source[start:i]
            value = float(text) if is_float else int(text)
            tokens.append(Token("NUMBER", value, line))
            continue

        # Identifiers / keywords ----------------------------------------
        if c.isalpha() or c == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            text = source[start:i]
            ttype = KEYWORDS.get(text, "IDENT")
            tokens.append(Token(ttype, text, line))
            continue

        # Two-character operators ---------------------------------------
        pair = source[i:i + 2]
        if pair in TWO_CHAR:
            tokens.append(Token(TWO_CHAR[pair], pair, line))
            i += 2
            continue

        # Single-character operators / punctuation ----------------------
        if c in ONE_CHAR:
            tokens.append(Token(ONE_CHAR[c], c, line))
            i += 1
            continue

        raise LexError("Unexpected character %r on line %d" % (c, line))

    tokens.append(Token("EOF", None, line))
    return tokens
