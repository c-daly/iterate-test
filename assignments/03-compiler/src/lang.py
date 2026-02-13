from lexer import Lexer
from parser import Parser
from compiler import Compiler
from vm import VM


def execute(source: str) -> list:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    return VM(code).run()
