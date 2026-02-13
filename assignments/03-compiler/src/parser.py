from dataclasses import dataclass
from typing import Any

from lexer import TokenType, Token


# AST Nodes
@dataclass
class NumberLit:
    value: float | int


@dataclass
class StringLit:
    value: str


@dataclass
class BoolLit:
    value: bool


@dataclass
class NilLit:
    pass


@dataclass
class Identifier:
    name: str


@dataclass
class UnaryOp:
    op: str  # '-' or 'not'
    operand: Any


@dataclass
class BinOp:
    op: str
    left: Any
    right: Any


@dataclass
class Assignment:
    name: str
    value: Any


@dataclass
class LetStmt:
    name: str
    value: Any


@dataclass
class IfStmt:
    condition: Any
    then_body: list
    else_body: list | None


@dataclass
class WhileStmt:
    condition: Any
    body: list


@dataclass
class PrintStmt:
    value: Any


@dataclass
class FnDecl:
    name: str
    params: list
    body: list


@dataclass
class ReturnStmt:
    value: Any


@dataclass
class CallExpr:
    callee: Any
    args: list


@dataclass
class ExprStmt:
    expr: Any


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> list:
        stmts = []
        while not self._at_end():
            stmts.append(self._statement())
        return stmts

    def _at_end(self):
        return self.tokens[self.pos].type == TokenType.EOF

    def _peek(self):
        return self.tokens[self.pos]

    def _advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tt: TokenType):
        tok = self._advance()
        if tok.type != tt:
            raise ParseError(f"Expected {tt}, got {tok.type} at line {tok.line}")
        return tok

    def _match(self, *types):
        if self._peek().type in types:
            return self._advance()
        return None

    # --- Statements ---

    def _statement(self):
        tt = self._peek().type
        if tt == TokenType.LET:
            return self._let_stmt()
        if tt == TokenType.IF:
            return self._if_stmt()
        if tt == TokenType.WHILE:
            return self._while_stmt()
        if tt == TokenType.PRINT:
            return self._print_stmt()
        if tt == TokenType.FN:
            return self._fn_decl()
        if tt == TokenType.RETURN:
            return self._return_stmt()
        return self._expr_stmt()

    def _let_stmt(self):
        self._expect(TokenType.LET)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.ASSIGN)
        value = self._expression()
        self._expect(TokenType.SEMI)
        return LetStmt(name=name_tok.value, value=value)

    def _if_stmt(self):
        self._expect(TokenType.IF)
        condition = self._expression()
        then_body = self._block()
        else_body = None
        if self._match(TokenType.ELSE):
            else_body = self._block()
        return IfStmt(condition=condition, then_body=then_body, else_body=else_body)

    def _while_stmt(self):
        self._expect(TokenType.WHILE)
        condition = self._expression()
        body = self._block()
        return WhileStmt(condition=condition, body=body)

    def _print_stmt(self):
        self._expect(TokenType.PRINT)
        value = self._expression()
        self._expect(TokenType.SEMI)
        return PrintStmt(value=value)

    def _fn_decl(self):
        self._expect(TokenType.FN)
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.LPAREN)
        params = []
        if self._peek().type != TokenType.RPAREN:
            params.append(self._expect(TokenType.IDENT).value)
            while self._match(TokenType.COMMA):
                params.append(self._expect(TokenType.IDENT).value)
        self._expect(TokenType.RPAREN)
        body = self._block()
        return FnDecl(name=name_tok.value, params=params, body=body)

    def _return_stmt(self):
        self._expect(TokenType.RETURN)
        value = None
        if self._peek().type != TokenType.SEMI:
            value = self._expression()
        self._expect(TokenType.SEMI)
        return ReturnStmt(value=value)

    def _block(self):
        self._expect(TokenType.LBRACE)
        stmts = []
        while self._peek().type != TokenType.RBRACE:
            stmts.append(self._statement())
        self._expect(TokenType.RBRACE)
        return stmts

    def _expr_stmt(self):
        expr = self._expression()
        self._expect(TokenType.SEMI)
        return ExprStmt(expr=expr)

    # --- Expressions (lowest to highest precedence) ---

    def _expression(self):
        return self._assignment()

    def _assignment(self):
        # Save position; try IDENT ASSIGN pattern
        saved = self.pos
        if self._peek().type == TokenType.IDENT:
            name_tok = self._advance()
            if self._match(TokenType.ASSIGN):
                value = self._expression()
                return Assignment(name=name_tok.value, value=value)
            # Not an assignment, restore position
            self.pos = saved
        return self._logic_or()

    def _logic_or(self):
        left = self._logic_and()
        while self._match(TokenType.OR):
            right = self._logic_and()
            left = BinOp(op="or", left=left, right=right)
        return left

    def _logic_and(self):
        left = self._comparison()
        while self._match(TokenType.AND):
            right = self._comparison()
            left = BinOp(op="and", left=left, right=right)
        return left

    def _comparison(self):
        left = self._addition()
        while tok := self._match(
            TokenType.EQ, TokenType.NE,
            TokenType.LT, TokenType.GT,
            TokenType.LE, TokenType.GE,
        ):
            right = self._addition()
            left = BinOp(op=tok.value, left=left, right=right)
        return left

    def _addition(self):
        left = self._multiply()
        while tok := self._match(TokenType.PLUS, TokenType.MINUS):
            right = self._multiply()
            left = BinOp(op=tok.value, left=left, right=right)
        return left

    def _multiply(self):
        left = self._unary()
        while tok := self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            right = self._unary()
            left = BinOp(op=tok.value, left=left, right=right)
        return left

    def _unary(self):
        if tok := self._match(TokenType.MINUS):
            operand = self._unary()
            return UnaryOp(op="-", operand=operand)
        if tok := self._match(TokenType.NOT):
            operand = self._unary()
            return UnaryOp(op="not", operand=operand)
        return self._call()

    def _call(self):
        expr = self._primary()
        while self._match(TokenType.LPAREN):
            args = []
            if self._peek().type != TokenType.RPAREN:
                args.append(self._expression())
                while self._match(TokenType.COMMA):
                    args.append(self._expression())
            self._expect(TokenType.RPAREN)
            expr = CallExpr(callee=expr, args=args)
        return expr

    def _primary(self):
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
            expr = self._expression()
            self._expect(TokenType.RPAREN)
            return expr

        raise ParseError(
            f"Unexpected token {tok.type} at line {tok.line}"
        )
