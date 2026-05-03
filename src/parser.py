"""Recursive-descent parser.

AST nodes are simple dataclasses. Each statement node represents a syntactic
construct from the grammar in the spec; expression nodes likewise.

Top-level result is Program(statements=[...]).
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .lexer import Token


# ---- AST node definitions -------------------------------------------------

@dataclass
class Program:
    statements: List[Any]


@dataclass
class Let:
    name: str
    value: Any


@dataclass
class Assign:
    name: str
    value: Any


@dataclass
class If:
    cond: Any
    then_block: List[Any]
    else_block: Optional[List[Any]] = None


@dataclass
class While:
    cond: Any
    body: List[Any]


@dataclass
class Print:
    value: Any


@dataclass
class FnDecl:
    name: str
    params: List[str]
    body: List[Any]


@dataclass
class Return:
    value: Any  # Optional[Any]; None means bare return


@dataclass
class ExprStmt:
    expr: Any


@dataclass
class Binary:
    op: str
    left: Any
    right: Any


@dataclass
class Unary:
    op: str
    operand: Any


@dataclass
class Logical:
    op: str
    left: Any
    right: Any


@dataclass
class Call:
    callee: Any
    args: List[Any] = field(default_factory=list)


@dataclass
class Number:
    value: Any


@dataclass
class String:
    value: str


@dataclass
class Bool:
    value: bool


@dataclass
class Nil:
    pass


@dataclass
class Ident:
    name: str


# ---- Parser ---------------------------------------------------------------

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # ---- helpers ---------------------------------------------------
    def _peek(self, off: int = 0) -> Token:
        return self.tokens[self.pos + off]

    def _check(self, *types: str) -> bool:
        return self._peek().type in types

    def _match(self, *types: str) -> bool:
        if self._check(*types):
            self.pos += 1
            return True
        return False

    def _consume(self, type_: str, msg: str = "") -> Token:
        if self._check(type_):
            tok = self._peek()
            self.pos += 1
            return tok
        actual = self._peek()
        raise ParseError("Expected " + type_ + " got " + str(actual.type) + " (" + msg + ")")

    # ---- top level ------------------------------------------------
    def parse(self) -> Program:
        stmts: List[Any] = []
        while not self._check("EOF"):
            stmts.append(self._statement())
        return Program(statements=stmts)

    # ---- statements -----------------------------------------------
    def _statement(self) -> Any:
        if self._check("let"):
            return self._let_stmt()
        if self._check("if"):
            return self._if_stmt()
        if self._check("while"):
            return self._while_stmt()
        if self._check("print"):
            return self._print_stmt()
        if self._check("fn"):
            return self._fn_decl()
        if self._check("return"):
            return self._return_stmt()
        return self._expr_stmt()

    def _let_stmt(self) -> Let:
        self._consume("let")
        name_tok = self._consume("IDENT", "variable name")
        self._consume("=", "after let name")
        value = self._expression()
        self._consume(";", "after let")
        return Let(name=name_tok.value, value=value)

    def _if_stmt(self) -> If:
        self._consume("if")
        cond = self._expression()
        then_block = self._block()
        else_block: Optional[List[Any]] = None
        if self._match("else"):
            else_block = self._block()
        return If(cond=cond, then_block=then_block, else_block=else_block)

    def _while_stmt(self) -> While:
        self._consume("while")
        cond = self._expression()
        body = self._block()
        return While(cond=cond, body=body)

    def _print_stmt(self) -> Print:
        self._consume("print")
        value = self._expression()
        self._consume(";", "after print expr")
        return Print(value=value)

    def _fn_decl(self) -> FnDecl:
        self._consume("fn")
        name_tok = self._consume("IDENT", "function name")
        self._consume("(")
        params: List[str] = []
        if not self._check(")"):
            params.append(self._consume("IDENT", "param").value)
            while self._match(","):
                params.append(self._consume("IDENT", "param").value)
        self._consume(")")
        body = self._block()
        return FnDecl(name=name_tok.value, params=params, body=body)

    def _return_stmt(self) -> Return:
        self._consume("return")
        value: Any = None
        if not self._check(";"):
            value = self._expression()
        self._consume(";", "after return")
        return Return(value=value)

    def _expr_stmt(self) -> ExprStmt:
        expr = self._expression()
        self._consume(";", "after expression statement")
        return ExprStmt(expr=expr)

    def _block(self) -> List[Any]:
        self._consume("{")
        stmts: List[Any] = []
        while not self._check("}") and not self._check("EOF"):
            stmts.append(self._statement())
        self._consume("}")
        return stmts

    # ---- expressions ----------------------------------------------
    def _expression(self) -> Any:
        return self._assignment()

    def _assignment(self) -> Any:
        # Look ahead for IDENT '=' (not '==')
        if self._check("IDENT") and self._peek(1).type == "=":
            name = self._peek().value
            self.pos += 2
            value = self._assignment()
            return Assign(name=name, value=value)
        return self._logic_or()

    def _logic_or(self) -> Any:
        left = self._logic_and()
        while self._check("or"):
            self.pos += 1
            right = self._logic_and()
            left = Logical(op="or", left=left, right=right)
        return left

    def _logic_and(self) -> Any:
        left = self._comparison()
        while self._check("and"):
            self.pos += 1
            right = self._comparison()
            left = Logical(op="and", left=left, right=right)
        return left

    def _comparison(self) -> Any:
        left = self._addition()
        while self._check("==", "!=", "<", ">", "<=", ">="):
            op = self._peek().type
            self.pos += 1
            right = self._addition()
            left = Binary(op=op, left=left, right=right)
        return left

    def _addition(self) -> Any:
        left = self._multiply()
        while self._check("+", "-"):
            op = self._peek().type
            self.pos += 1
            right = self._multiply()
            left = Binary(op=op, left=left, right=right)
        return left

    def _multiply(self) -> Any:
        left = self._unary()
        while self._check("*", "/", "%"):
            op = self._peek().type
            self.pos += 1
            right = self._unary()
            left = Binary(op=op, left=left, right=right)
        return left

    def _unary(self) -> Any:
        if self._check("-", "not"):
            op = self._peek().type
            self.pos += 1
            operand = self._unary()
            return Unary(op=op, operand=operand)
        return self._call()

    def _call(self) -> Any:
        expr = self._primary()
        while self._match("("):
            args: List[Any] = []
            if not self._check(")"):
                args.append(self._expression())
                while self._match(","):
                    args.append(self._expression())
            self._consume(")")
            expr = Call(callee=expr, args=args)
        return expr

    def _primary(self) -> Any:
        tok = self._peek()
        if tok.type == "NUMBER":
            self.pos += 1
            return Number(value=tok.value)
        if tok.type == "STRING":
            self.pos += 1
            return String(value=tok.value)
        if tok.type == "true":
            self.pos += 1
            return Bool(value=True)
        if tok.type == "false":
            self.pos += 1
            return Bool(value=False)
        if tok.type == "nil":
            self.pos += 1
            return Nil()
        if tok.type == "IDENT":
            self.pos += 1
            return Ident(name=tok.value)
        if tok.type == "(":
            self.pos += 1
            expr = self._expression()
            self._consume(")")
            return expr
        raise ParseError("Unexpected token " + str(tok.type) + " at line " + str(tok.line))


def parse(tokens: List[Token]) -> Program:
    return Parser(tokens).parse()
