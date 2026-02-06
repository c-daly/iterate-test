"""Convenience function: execute source and return printed output."""

from src.lexer import tokenize
from src.parser import parse
from src.compiler import compile_ast
from src.vm import run


def execute(source: str) -> list[str]:
    """Tokenize, parse, compile, and execute source code.

    Returns a list of strings that were printed during execution.
    """
    tokens = tokenize(source)
    ast = parse(tokens)
    program = compile_ast(ast)
    return run(program)
