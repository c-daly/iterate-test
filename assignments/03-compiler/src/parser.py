"""Recursive-descent parser: token stream -> AST.

AST nodes are plain dataclass-like objects. The compiler walks them. The
grammar implemented here matches the one in the assignment README, with the
usual precedence climbing for expressions:

    assignment -> logic_or -> logic_and -> comparison
    -> addition -> multiply -> unary -> call -> primary
"""

from src.lexer import tokenize


class ParseError(Exception):
    """Raised on malformed input that does not match the grammar."""


# --- AST node types --------------------------------------------------------

class Node:
    pass


class Program(Node):
    def __init__(self, statements):
        self.statements = statements


class LetStmt(Node):
    def __init__(self, name, value):
        self.name = name
        self.value = value


class IfStmt(Node):
    def __init__(self, condition, then_block, else_block):
        self.condition = condition
        self.then_block = then_block      # list of statements
        self.else_block = else_block      # list of statements or None


class WhileStmt(Node):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body                  # list of statements


class PrintStmt(Node):
    def __init__(self, value):
        self.value = value


class FnDecl(Node):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params              # list of str
        self.body = body                  # list of statements


class ReturnStmt(Node):
    def __init__(self, value):
        self.value = value                # expr or None


class ExprStmt(Node):
    def __init__(self, expr):
        self.expr = expr


class Assign(Node):
    def __init__(self, name, value):
        self.name = name
        self.value = value


class Binary(Node):
    def __init__(self, op, left, right):
        self.op = op                      # token type string, e.g. "PLUS"
        self.left = left
        self.right = right


class Logical(Node):
    def __init__(self, op, left, right):
        self.op = op                      # "AND" or "OR"
        self.left = left
        self.right = right


class Unary(Node):
    def __init__(self, op, operand):
        self.op = op                      # "MINUS" or "NOT"
        self.operand = operand


class Call(Node):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args                  # list of expr


class Literal(Node):
    def __init__(self, value):
        self.value = value


class Variable(Node):
    def __init__(self, name):
        self.name = name


# --- Parser ----------------------------------------------------------------

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # token helpers ---------------------------------------------------------
    def peek(self):
        return self.tokens[self.pos]

    def previous(self):
        return self.tokens[self.pos - 1]

    def check(self, type_):
        return self.peek().type == type_

    def at_end(self):
        return self.peek().type == "EOF"

    def advance(self):
        if not self.at_end():
            self.pos += 1
        return self.previous()

    def match(self, *types):
        if self.peek().type in types:
            return self.advance()
        return None

    def expect(self, type_, message):
        if self.check(type_):
            return self.advance()
        tok = self.peek()
        raise ParseError("%s (got %r on line %d)" % (message, tok.value, tok.line))

    # entry -----------------------------------------------------------------
    def parse(self):
        statements = []
        while not self.at_end():
            statements.append(self.statement())
        return Program(statements)

    # statements ------------------------------------------------------------
    def statement(self):
        tok = self.peek()
        if tok.type == "LET":
            return self.let_stmt()
        if tok.type == "IF":
            return self.if_stmt()
        if tok.type == "WHILE":
            return self.while_stmt()
        if tok.type == "PRINT":
            return self.print_stmt()
        if tok.type == "FN":
            return self.fn_decl()
        if tok.type == "RETURN":
            return self.return_stmt()
        return self.expr_stmt()

    def let_stmt(self):
        self.expect("LET", "expected let")
        name = self.expect("IDENT", "expected variable name after let").value
        self.expect("ASSIGN", "expected = in let statement")
        value = self.expression()
        self.expect("SEMI", "expected ; after let statement")
        return LetStmt(name, value)

    def if_stmt(self):
        self.expect("IF", "expected if")
        condition = self.expression()
        then_block = self.block()
        else_block = None
        if self.match("ELSE"):
            else_block = self.block()
        return IfStmt(condition, then_block, else_block)

    def while_stmt(self):
        self.expect("WHILE", "expected while")
        condition = self.expression()
        body = self.block()
        return WhileStmt(condition, body)

    def print_stmt(self):
        self.expect("PRINT", "expected print")
        value = self.expression()
        self.expect("SEMI", "expected ; after print statement")
        return PrintStmt(value)

    def fn_decl(self):
        self.expect("FN", "expected fn")
        name = self.expect("IDENT", "expected function name").value
        self.expect("LPAREN", "expected ( after function name")
        params = []
        if not self.check("RPAREN"):
            params.append(self.expect("IDENT", "expected parameter name").value)
            while self.match("COMMA"):
                params.append(self.expect("IDENT", "expected parameter name").value)
        self.expect("RPAREN", "expected ) after parameters")
        body = self.block()
        return FnDecl(name, params, body)

    def return_stmt(self):
        self.expect("RETURN", "expected return")
        value = None
        if not self.check("SEMI"):
            value = self.expression()
        self.expect("SEMI", "expected ; after return statement")
        return ReturnStmt(value)

    def expr_stmt(self):
        expr = self.expression()
        self.expect("SEMI", "expected ; after expression statement")
        return ExprStmt(expr)

    def block(self):
        self.expect("LBRACE", "expected { to start block")
        statements = []
        while not self.check("RBRACE") and not self.at_end():
            statements.append(self.statement())
        self.expect("RBRACE", "expected } to close block")
        return statements

    # expressions -----------------------------------------------------------
    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.logic_or()
        if self.match("ASSIGN"):
            value = self.assignment()
            if isinstance(expr, Variable):
                return Assign(expr.name, value)
            raise ParseError("invalid assignment target")
        return expr

    def logic_or(self):
        expr = self.logic_and()
        while self.match("OR"):
            right = self.logic_and()
            expr = Logical("OR", expr, right)
        return expr

    def logic_and(self):
        expr = self.comparison()
        while self.match("AND"):
            right = self.comparison()
            expr = Logical("AND", expr, right)
        return expr

    def comparison(self):
        expr = self.addition()
        while self.peek().type in ("EQ", "NE", "LT", "GT", "LE", "GE"):
            op = self.advance().type
            right = self.addition()
            expr = Binary(op, expr, right)
        return expr

    def addition(self):
        expr = self.multiply()
        while self.peek().type in ("PLUS", "MINUS"):
            op = self.advance().type
            right = self.multiply()
            expr = Binary(op, expr, right)
        return expr

    def multiply(self):
        expr = self.unary()
        while self.peek().type in ("STAR", "SLASH", "PERCENT"):
            op = self.advance().type
            right = self.unary()
            expr = Binary(op, expr, right)
        return expr

    def unary(self):
        if self.peek().type in ("MINUS", "NOT"):
            op = self.advance().type
            operand = self.unary()
            return Unary(op, operand)
        return self.call()

    def call(self):
        expr = self.primary()
        while self.match("LPAREN"):
            args = []
            if not self.check("RPAREN"):
                args.append(self.expression())
                while self.match("COMMA"):
                    args.append(self.expression())
            self.expect("RPAREN", "expected ) after arguments")
            expr = Call(expr, args)
        return expr

    def primary(self):
        tok = self.peek()
        if tok.type == "NUMBER":
            self.advance()
            return Literal(tok.value)
        if tok.type == "STRING":
            self.advance()
            return Literal(tok.value)
        if tok.type == "TRUE":
            self.advance()
            return Literal(True)
        if tok.type == "FALSE":
            self.advance()
            return Literal(False)
        if tok.type == "NIL":
            self.advance()
            return Literal(None)
        if tok.type == "IDENT":
            self.advance()
            return Variable(tok.value)
        if tok.type == "LPAREN":
            self.advance()
            expr = self.expression()
            self.expect("RPAREN", "expected ) after expression")
            return expr
        raise ParseError("unexpected token %r on line %d" % (tok.value, tok.line))


def parse(source):
    """Convenience: tokenize then parse a source string into a Program."""
    tokens = tokenize(source)
    return Parser(tokens).parse()
