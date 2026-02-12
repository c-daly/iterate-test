"""Convenience function to execute source code."""

from typing import List
from src.lexer import tokenize
from src.parser import Parser
from src.compiler import Compiler
from src.vm import VM


def execute(source: str) -> List[str]:
    """Tokenize, parse, compile, and execute source code. Returns list of printed output."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    ast = parser.parse()
    compiler = Compiler()
    code = compiler.compile(ast)
    vm = VM(code, compiler.functions)
    return vm.run()
