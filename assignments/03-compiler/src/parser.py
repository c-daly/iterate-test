"""Parser: parse token stream into AST nodes."""

from dataclasses import dataclass, field
from src.lexer import Token, TokenType


# AST node base
@dataclass
class ASTNode:
    pass


# Expressions
@dataclass
class NumberLit(ASTNode):
    value: float


@dataclass
class StringLit(ASTNode):
    value: str


@dataclass
class BoolLit(ASTNode):
    value: bool


@dataclass
class NilLit(ASTNode):
    pass


@dataclass
class Identifier(ASTNode):
    name: str


@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode


@dataclass
class BinaryOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


@dataclass
class Assignment(ASTNode):
    name: str
    value: ASTNode


@dataclass
class CallExpr(ASTNode):
    callee: ASTNode
    args: list[ASTNode]


# Statements
@dataclass
class LetStmt(ASTNode):
    name: str
    value: ASTNode


@dataclass
class PrintStmt(ASTNode):
    value: ASTNode


@dataclass
class ExprStmt(ASTNode):
    expr: ASTNode


@dataclass
class Block(ASTNode):
    stmts: list[ASTNode] = field(default_factory=list)


@dataclass
class IfStmt(ASTNode):
    condition: ASTNode
    then_block: Block
    else_block: Block | None = None


@dataclass
class WhileStmt(ASTNode):
    condition: ASTNode
    body: Block


@dataclass
class FnDecl(ASTNode):
    name: str
    params: list[str]
    body: Block


@dataclass
class ReturnStmt(ASTNode):
    value: ASTNode | None = None


@dataclass
class Program(ASTNode):
    stmts: list[ASTNode] = field(default_factory=list)


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def check(self, tt: TokenType) -> bool:
        return self.peek().type == tt

    def match(self, *types: TokenType) -> Token | None:
        if self.peek().type in types:
            return self.advance()
        return None

    def expect(self, tt: TokenType) -> Token:
        tok = self.advance()
        if tok.type != tt:
            raise SyntaxError(
                f"Expected {tt.name}, got {tok.type.name} "
                f"({tok.value!r}) at line {tok.line}"
            )
        return tok

    def parse_program(self) -> Program:
        stmts = []
        while not self.check(TokenType.EOF):
            stmts.append(self.parse_statement())
        return Program(stmts)

    def parse_statement(self):
        if self.check(TokenType.LET):
            return self.parse_let()
        if self.check(TokenType.IF):
            return self.parse_if()
        if self.check(TokenType.WHILE):
            return self.parse_while()
        if self.check(TokenType.PRINT):
            return self.parse_print()
        if self.check(TokenType.FN):
            return self.parse_fn()
        if self.check(TokenType.RETURN):
            return self.parse_return()
        return self.parse_expr_stmt()

    def parse_let(self) -> LetStmt:
        self.advance()  # consume 'let'
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expr()
        self.expect(TokenType.SEMICOLON)
        return LetStmt(name, value)

    def parse_if(self) -> IfStmt:
        self.advance()  # consume 'if'
        condition = self.parse_expr()
        then_block = self.parse_block()
        else_block = None
        if self.match(TokenType.ELSE):
            else_block = self.parse_block()
        return IfStmt(condition, then_block, else_block)

    def parse_while(self) -> WhileStmt:
        self.advance()  # consume 'while'
        condition = self.parse_expr()
        body = self.parse_block()
        return WhileStmt(condition, body)

    def parse_print(self) -> PrintStmt:
        self.advance()  # consume 'print'
        value = self.parse_expr()
        self.expect(TokenType.SEMICOLON)
        return PrintStmt(value)

    def parse_fn(self) -> FnDecl:
        self.advance()  # consume 'fn'
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        params = []
        if not self.check(TokenType.RPAREN):
            params.append(self.expect(TokenType.IDENT).value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return FnDecl(name, params, body)

    def parse_return(self) -> ReturnStmt:
        self.advance()  # consume 'return'
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.parse_expr()
        self.expect(TokenType.SEMICOLON)
        return ReturnStmt(value)

    def parse_expr_stmt(self) -> ExprStmt:
        expr = self.parse_expr()
        self.expect(TokenType.SEMICOLON)
        return ExprStmt(expr)

    def parse_block(self) -> Block:
        self.expect(TokenType.LBRACE)
        stmts = []
        while not self.check(TokenType.RBRACE):
            stmts.append(self.parse_statement())
        self.expect(TokenType.RBRACE)
        return Block(stmts)

    def parse_expr(self):
        return self.parse_assignment()

    def parse_assignment(self):
        # If we see IDENT followed by '=', it's an assignment
        if self.check(TokenType.IDENT):
            # Lookahead
            if (self.pos + 1 < len(self.tokens)
                    and self.tokens[self.pos + 1].type == TokenType.ASSIGN):
                name = self.advance().value
                self.advance()  # consume '='
                value = self.parse_expr()
                return Assignment(name, value)
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.match(TokenType.OR):
            right = self.parse_and()
            left = BinaryOp("or", left, right)
        return left

    def parse_and(self):
        left = self.parse_comparison()
        while self.match(TokenType.AND):
            right = self.parse_comparison()
            left = BinaryOp("and", left, right)
        return left

    def parse_comparison(self):
        left = self.parse_addition()
        op_map = {
            TokenType.EQ: "==",
            TokenType.NE: "!=",
            TokenType.LT: "<",
            TokenType.GT: ">",
            TokenType.LE: "<=",
            TokenType.GE: ">=",
        }
        while self.peek().type in op_map:
            tok = self.advance()
            right = self.parse_addition()
            left = BinaryOp(op_map[tok.type], left, right)
        return left

    def parse_addition(self):
        left = self.parse_multiply()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            tok = self.advance()
            op = "+" if tok.type == TokenType.PLUS else "-"
            right = self.parse_multiply()
            left = BinaryOp(op, left, right)
        return left

    def parse_multiply(self):
        left = self.parse_unary()
        while self.peek().type in (
            TokenType.STAR, TokenType.SLASH, TokenType.PERCENT
        ):
            tok = self.advance()
            op_map = {
                TokenType.STAR: "*",
                TokenType.SLASH: "/",
                TokenType.PERCENT: "%",
            }
            right = self.parse_unary()
            left = BinaryOp(op_map[tok.type], left, right)
        return left

    def parse_unary(self):
        if self.match(TokenType.MINUS):
            operand = self.parse_unary()
            return UnaryOp("-", operand)
        if self.match(TokenType.NOT):
            operand = self.parse_unary()
            return UnaryOp("not", operand)
        return self.parse_call()

    def parse_call(self):
        expr = self.parse_primary()
        while self.match(TokenType.LPAREN):
            args = []
            if not self.check(TokenType.RPAREN):
                args.append(self.parse_expr())
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expr())
            self.expect(TokenType.RPAREN)
            expr = CallExpr(expr, args)
        return expr

    def parse_primary(self):
        tok = self.peek()

        if tok.type == TokenType.NUMBER:
            self.advance()
            return NumberLit(tok.value)

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
            return Identifier(tok.value)

        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        raise SyntaxError(
            f"Unexpected token {tok.type.name} ({tok.value!r}) "
            f"at line {tok.line}"
        )


def parse(tokens) -> Program:
    parser = Parser(tokens)
    return parser.parse_program()
