"""Recursive-descent parser producing AST nodes.

Grammar mirrors README spec: program, statements, expressions with the
standard precedence ladder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .lexer import Token, TokenType


class ParseError(Exception):
    """Raised on a syntactic error."""


# ---------------------------------------------------------------- AST nodes
@dataclass
class Node:
    pass


@dataclass
class Program(Node):
    statements: List[Node] = field(default_factory=list)


@dataclass
class LetStmt(Node):
    name: str
    value: Node


@dataclass
class AssignStmt(Node):
    name: str
    value: Node


@dataclass
class IfStmt(Node):
    cond: Node
    then_block: List[Node]
    else_block: Optional[List[Node]]


@dataclass
class WhileStmt(Node):
    cond: Node
    body: List[Node]


@dataclass
class PrintStmt(Node):
    expr: Node


@dataclass
class ReturnStmt(Node):
    expr: Optional[Node]


@dataclass
class FnDecl(Node):
    name: str
    params: List[str]
    body: List[Node]


@dataclass
class ExprStmt(Node):
    expr: Node


@dataclass
class NumberLit(Node):
    value: Any


@dataclass
class StringLit(Node):
    value: str


@dataclass
class BoolLit(Node):
    value: bool


@dataclass
class NilLit(Node):
    pass


@dataclass
class Identifier(Node):
    name: str


@dataclass
class Unary(Node):
    op: str
    operand: Node


@dataclass
class Binary(Node):
    op: str
    left: Node
    right: Node


@dataclass
class Logical(Node):
    op: str  # "and" or "or"
    left: Node
    right: Node


@dataclass
class Call(Node):
    callee: Node
    args: List[Node]


@dataclass
class Assign(Node):
    name: str
    value: Node


# ---------------------------------------------------------------- Parser
class Parser:
    """Recursive descent parser.

    Consumes a token stream produced by Lexer.tokenize() and returns a
    Program node.
    """

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Program:
        stmts: List[Node] = []
        while not self._at_end():
            stmts.append(self._statement())
        return Program(statements=stmts)

    # ----------------------------------------------------------- helpers
    def _at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        if not self._at_end():
            self.pos += 1
        return tok

    def _check(self, kind: TokenType) -> bool:
        return self._peek().type == kind

    def _match(self, *kinds: TokenType) -> bool:
        if self._peek().type in kinds:
            self._advance()
            return True
        return False

    def _expect(self, kind: TokenType, msg: str) -> Token:
        if self._check(kind):
            return self._advance()
        tok = self._peek()
        raise ParseError(f"Line {tok.line}: expected {msg}, got {tok.type.name}")

    # ----------------------------------------------------------- statements
    def _statement(self) -> Node:
        tok = self._peek()
        if tok.type == TokenType.LET:
            return self._let_stmt()
        if tok.type == TokenType.IF:
            return self._if_stmt()
        if tok.type == TokenType.WHILE:
            return self._while_stmt()
        if tok.type == TokenType.PRINT:
            return self._print_stmt()
        if tok.type == TokenType.FN:
            return self._fn_decl()
        if tok.type == TokenType.RETURN:
            return self._return_stmt()
        return self._expr_stmt()

    def _let_stmt(self) -> Node:
        self._advance()  # let
        name_tok = self._expect(TokenType.IDENT, "identifier")
        self._expect(TokenType.ASSIGN, "=")
        value = self._expr()
        self._expect(TokenType.SEMI, ";")
        return LetStmt(name=name_tok.value, value=value)

    def _if_stmt(self) -> Node:
        self._advance()  # if
        cond = self._expr()
        then_block = self._block()
        else_block: Optional[List[Node]] = None
        if self._match(TokenType.ELSE):
            else_block = self._block()
        return IfStmt(cond=cond, then_block=then_block, else_block=else_block)

    def _while_stmt(self) -> Node:
        self._advance()  # while
        cond = self._expr()
        body = self._block()
        return WhileStmt(cond=cond, body=body)

    def _print_stmt(self) -> Node:
        self._advance()  # print
        expr = self._expr()
        self._expect(TokenType.SEMI, ";")
        return PrintStmt(expr=expr)

    def _fn_decl(self) -> Node:
        self._advance()  # fn
        name_tok = self._expect(TokenType.IDENT, "function name")
        self._expect(TokenType.LPAREN, "(")
        params: List[str] = []
        if not self._check(TokenType.RPAREN):
            params.append(self._expect(TokenType.IDENT, "parameter name").value)
            while self._match(TokenType.COMMA):
                params.append(self._expect(TokenType.IDENT, "parameter name").value)
        self._expect(TokenType.RPAREN, ")")
        body = self._block()
        return FnDecl(name=name_tok.value, params=params, body=body)

    def _return_stmt(self) -> Node:
        self._advance()  # return
        expr: Optional[Node] = None
        if not self._check(TokenType.SEMI):
            expr = self._expr()
        self._expect(TokenType.SEMI, ";")
        return ReturnStmt(expr=expr)

    def _expr_stmt(self) -> Node:
        expr = self._expr()
        self._expect(TokenType.SEMI, ";")
        return ExprStmt(expr=expr)

    def _block(self) -> List[Node]:
        self._expect(TokenType.LBRACE, "{")
        stmts: List[Node] = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            stmts.append(self._statement())
        self._expect(TokenType.RBRACE, "}")
        return stmts

    # ----------------------------------------------------------- expressions
    def _expr(self) -> Node:
        return self._assignment()

    def _assignment(self) -> Node:
        # IDENT = expr  -- otherwise fall back to logic_or
        if self._check(TokenType.IDENT) and self._peek(1).type == TokenType.ASSIGN:
            name = self._advance().value
            self._advance()  # =
            value = self._assignment()
            return Assign(name=name, value=value)
        return self._logic_or()

    def _logic_or(self) -> Node:
        left = self._logic_and()
        while self._match(TokenType.OR):
            right = self._logic_and()
            left = Logical(op="or", left=left, right=right)
        return left

    def _logic_and(self) -> Node:
        left = self._comparison()
        while self._match(TokenType.AND):
            right = self._comparison()
            left = Logical(op="and", left=left, right=right)
        return left

    _CMP_OPS = {
        TokenType.EQ: "==",
        TokenType.NE: "!=",
        TokenType.LT: "<",
        TokenType.GT: ">",
        TokenType.LE: "<=",
        TokenType.GE: ">=",
    }

    def _comparison(self) -> Node:
        left = self._addition()
        while self._peek().type in self._CMP_OPS:
            tok = self._advance()
            right = self._addition()
            left = Binary(op=self._CMP_OPS[tok.type], left=left, right=right)
        return left

    _ADD_OPS = {TokenType.PLUS: "+", TokenType.MINUS: "-"}

    def _addition(self) -> Node:
        left = self._multiply()
        while self._peek().type in self._ADD_OPS:
            tok = self._advance()
            right = self._multiply()
            left = Binary(op=self._ADD_OPS[tok.type], left=left, right=right)
        return left

    _MUL_OPS = {TokenType.STAR: "*", TokenType.SLASH: "/", TokenType.PERCENT: "%"}

    def _multiply(self) -> Node:
        left = self._unary()
        while self._peek().type in self._MUL_OPS:
            tok = self._advance()
            right = self._unary()
            left = Binary(op=self._MUL_OPS[tok.type], left=left, right=right)
        return left

    def _unary(self) -> Node:
        if self._match(TokenType.MINUS):
            return Unary(op="-", operand=self._unary())
        if self._match(TokenType.NOT):
            return Unary(op="not", operand=self._unary())
        return self._call()

    def _call(self) -> Node:
        expr = self._primary()
        while True:
            if self._match(TokenType.LPAREN):
                args: List[Node] = []
                if not self._check(TokenType.RPAREN):
                    args.append(self._expr())
                    while self._match(TokenType.COMMA):
                        args.append(self._expr())
                self._expect(TokenType.RPAREN, ")")
                expr = Call(callee=expr, args=args)
            else:
                break
        return expr

    def _primary(self) -> Node:
        tok = self._peek()
        if tok.type == TokenType.NUMBER:
            self._advance()
            return NumberLit(value=tok.value)
        if tok.type == TokenType.STRING:
            self._advance()
            return StringLit(value=tok.value)
        if tok.type == TokenType.TRUE:
            self._advance()
            return BoolLit(value=True)
        if tok.type == TokenType.FALSE:
            self._advance()
            return BoolLit(value=False)
        if tok.type == TokenType.NIL:
            self._advance()
            return NilLit()
        if tok.type == TokenType.IDENT:
            self._advance()
            return Identifier(name=tok.value)
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._expr()
            self._expect(TokenType.RPAREN, ")")
            return expr
        raise ParseError(f"Line {tok.line}: unexpected token {tok.type.name}")
