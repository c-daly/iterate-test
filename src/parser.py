"""Parser - parses token stream into AST nodes."""

from dataclasses import dataclass, field
from typing import List, Optional
from src.lexer import Token, TokenType, LexerError


# --- AST Nodes ---

@dataclass
class NumberLit:
    value: float

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
class Ident:
    name: str

@dataclass
class UnaryOp:
    op: str
    operand: object

@dataclass
class BinOp:
    op: str
    left: object
    right: object

@dataclass
class Assign:
    name: str
    value: object

@dataclass
class Call:
    callee: object
    args: list

@dataclass
class LetStmt:
    name: str
    value: object

@dataclass
class PrintStmt:
    value: object

@dataclass
class ExprStmt:
    expr: object

@dataclass
class Block:
    stmts: list

@dataclass
class IfStmt:
    condition: object
    then_block: object
    else_block: object = None

@dataclass
class WhileStmt:
    condition: object
    body: object

@dataclass
class FnDecl:
    name: str
    params: list
    body: object

@dataclass
class ReturnStmt:
    value: object = None

@dataclass
class Program:
    stmts: list


# --- Parser ---

class ParseError(Exception):
    def __init__(self, message: str, token: Token = None):
        loc = f" at {token.line}:{token.col}" if token else ""
        super().__init__(f"Parse error{loc}: {message}")
        self.token = token


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def check(self, *types: TokenType) -> bool:
        return self.peek().type in types

    def match(self, *types: TokenType) -> Optional[Token]:
        if self.peek().type in types:
            return self.advance()
        return None

    def expect(self, tt: TokenType, msg: str = "") -> Token:
        tok = self.advance()
        if tok.type != tt:
            raise ParseError(msg or f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})", tok)
        return tok

    def parse(self) -> Program:
        stmts = []
        while not self.check(TokenType.EOF):
            stmts.append(self.statement())
        return Program(stmts)

    def statement(self):
        if self.check(TokenType.LET):
            return self.let_stmt()
        if self.check(TokenType.IF):
            return self.if_stmt()
        if self.check(TokenType.WHILE):
            return self.while_stmt()
        if self.check(TokenType.PRINT):
            return self.print_stmt()
        if self.check(TokenType.FN):
            return self.fn_decl()
        if self.check(TokenType.RETURN):
            return self.return_stmt()
        return self.expr_stmt()

    def let_stmt(self):
        self.advance()  # consume 'let'
        name_tok = self.expect(TokenType.IDENT, "Expected variable name after 'let'")
        self.expect(TokenType.EQ, "Expected '=' after variable name")
        value = self.expression()
        self.expect(TokenType.SEMICOLON, "Expected ';' after let statement")
        return LetStmt(name_tok.value, value)

    def if_stmt(self):
        self.advance()  # consume 'if'
        cond = self.expression()
        then_block = self.block()
        else_block = None
        if self.match(TokenType.ELSE):
            if self.check(TokenType.IF):
                else_block = Block([self.if_stmt()])
            else:
                else_block = self.block()
        return IfStmt(cond, then_block, else_block)

    def while_stmt(self):
        self.advance()  # consume 'while'
        cond = self.expression()
        body = self.block()
        return WhileStmt(cond, body)

    def print_stmt(self):
        self.advance()  # consume 'print'
        value = self.expression()
        self.expect(TokenType.SEMICOLON, "Expected ';' after print statement")
        return PrintStmt(value)

    def fn_decl(self):
        self.advance()  # consume 'fn'
        name_tok = self.expect(TokenType.IDENT, "Expected function name")
        self.expect(TokenType.LPAREN, "Expected '(' after function name")
        params = []
        if not self.check(TokenType.RPAREN):
            params.append(self.expect(TokenType.IDENT, "Expected parameter name").value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENT, "Expected parameter name").value)
        self.expect(TokenType.RPAREN, "Expected ')' after parameters")
        body = self.block()
        return FnDecl(name_tok.value, params, body)

    def return_stmt(self):
        self.advance()  # consume 'return'
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.expression()
        self.expect(TokenType.SEMICOLON, "Expected ';' after return statement")
        return ReturnStmt(value)

    def expr_stmt(self):
        expr = self.expression()
        self.expect(TokenType.SEMICOLON, "Expected ';' after expression")
        return ExprStmt(expr)

    def block(self):
        self.expect(TokenType.LBRACE, "Expected '{'")
        stmts = []
        while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
            stmts.append(self.statement())
        self.expect(TokenType.RBRACE, "Expected '}'")
        return Block(stmts)

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.logic_or()
        if self.match(TokenType.EQ):
            if isinstance(expr, Ident):
                value = self.assignment()
                return Assign(expr.name, value)
            raise ParseError("Invalid assignment target", self.peek())
        return expr

    def logic_or(self):
        left = self.logic_and()
        while self.match(TokenType.OR):
            right = self.logic_and()
            left = BinOp("or", left, right)
        return left

    def logic_and(self):
        left = self.comparison()
        while self.match(TokenType.AND):
            right = self.comparison()
            left = BinOp("and", left, right)
        return left

    def comparison(self):
        left = self.addition()
        while self.check(TokenType.EQ_EQ, TokenType.BANG_EQ, TokenType.LT, TokenType.GT, TokenType.LT_EQ, TokenType.GT_EQ):
            op_tok = self.advance()
            right = self.addition()
            left = BinOp(op_tok.value, left, right)
        return left

    def addition(self):
        left = self.multiply()
        while self.check(TokenType.PLUS, TokenType.MINUS):
            op_tok = self.advance()
            right = self.multiply()
            left = BinOp(op_tok.value, left, right)
        return left

    def multiply(self):
        left = self.unary()
        while self.check(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op_tok = self.advance()
            right = self.unary()
            left = BinOp(op_tok.value, left, right)
        return left

    def unary(self):
        if self.check(TokenType.MINUS):
            self.advance()
            operand = self.unary()
            return UnaryOp("-", operand)
        if self.check(TokenType.NOT):
            self.advance()
            operand = self.unary()
            return UnaryOp("not", operand)
        return self.call()

    def call(self):
        expr = self.primary()
        while self.check(TokenType.LPAREN):
            self.advance()  # consume '('
            args = []
            if not self.check(TokenType.RPAREN):
                args.append(self.expression())
                while self.match(TokenType.COMMA):
                    args.append(self.expression())
            self.expect(TokenType.RPAREN, "Expected ')' after arguments")
            expr = Call(expr, args)
        return expr

    def primary(self):
        tok = self.peek()

        if tok.type == TokenType.NUMBER:
            self.advance()
            val = float(tok.value) if "." in tok.value else int(tok.value)
            return NumberLit(val)

        if tok.type == TokenType.STRING:
            self.advance()
            return StringLit(tok.value)

        if tok.type == TokenType.TRUE:
            self.advance()
            return BoolLit(True)

        if tok.type == TokenType.FALSE:
            self.advance()
            return BoolLit(False)

        if tok.type == TokenType.NIL:
            self.advance()
            return NilLit()

        if tok.type == TokenType.IDENT:
            self.advance()
            return Ident(tok.value)

        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.expression()
            self.expect(TokenType.RPAREN, "Expected ')'")
            return expr

        raise ParseError(f"Unexpected token: {tok.type.name} ({tok.value!r})", tok)
