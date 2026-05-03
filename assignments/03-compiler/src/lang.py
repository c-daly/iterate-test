"""High-level convenience: compile + run a source string and return printed lines."""
from __future__ import annotations

from typing import List

from .compiler import Compiler
from .lexer import Lexer
from .parser import Parser
from .vm import VM


def execute(source: str) -> List[str]:
    """Compile then run the source. Returns the list of printed lines."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    chunk = Compiler().compile(ast)
    return VM().run(chunk)
