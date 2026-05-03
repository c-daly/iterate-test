"""Front-end convenience: source string -> printed output list."""
from typing import List

from .lexer import tokenize
from .parser import parse
from .compiler import compile_program
from .vm import run


def execute(source: str) -> List[str]:
    tokens = tokenize(source)
    ast = parse(tokens)
    code = compile_program(ast)
    return run(code)
