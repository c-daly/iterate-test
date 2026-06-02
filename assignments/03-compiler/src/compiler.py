"""Compiler: AST -> bytecode for the stack VM.

Bytecode model
--------------
A *chunk* is a flat list of instructions. Each instruction is a tuple whose
first element is an opcode string from `OPS` and whose remaining elements are
operands (a constant value, a variable name, or a jump target index).

Functions compile to their own `Function` object (name, params, chunk). A
function is itself a constant: `fn f(...) {...}` emits CONST <Function> then
DEFINE f, so the function value lives in a variable like any other binding.
CALL pops the callee and its arguments off the stack.

Scoping is resolved at *runtime* by the VM via a scope chain, so the compiler
only needs to emit name-based LOAD / DEFINE / STORE. DEFINE creates a binding
in the current (innermost) scope (used by `let` and parameters); STORE updates
the nearest existing binding (used by `x = expr`); LOAD reads through the
chain. This gives functions their own scope while still being able to read
outer/global variables.
"""

from src import parser as P

# Canonical opcode set required by the assignment, plus the two store/define
# variants and a JUMP_IF_TRUE helper used for short-circuit logic.
OPS = {
    "CONST", "POP", "ADD", "SUB", "MUL", "DIV", "MOD", "NEG", "NOT",
    "EQ", "NE", "LT", "GT", "LE", "GE", "AND", "OR",
    "LOAD", "STORE", "DEFINE", "JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE",
    "CALL", "RETURN", "PRINT", "HALT", "DUP",
}

BINARY_OP = {
    "PLUS": "ADD",
    "MINUS": "SUB",
    "STAR": "MUL",
    "SLASH": "DIV",
    "PERCENT": "MOD",
    "EQ": "EQ",
    "NE": "NE",
    "LT": "LT",
    "GT": "GT",
    "LE": "LE",
    "GE": "GE",
}


class CompileError(Exception):
    pass


class Function:
    """A compiled function: its name, parameter list, and bytecode chunk."""

    def __init__(self, name, params, chunk):
        self.name = name
        self.params = params
        self.arity = len(params)
        self.chunk = chunk

    def __repr__(self):
        return "<fn %s/%d>" % (self.name, self.arity)


class Chunk:
    """A growable list of instructions with simple patch support for jumps."""

    def __init__(self):
        self.code = []

    def emit(self, op, *operands):
        assert op in OPS, "unknown opcode %r" % op
        self.code.append((op,) + operands)
        return len(self.code) - 1

    def emit_jump(self, op):
        # Placeholder target -1, patched later to the real index.
        return self.emit(op, -1)

    def patch_jump(self, index):
        op = self.code[index][0]
        self.code[index] = (op, len(self.code))

    def here(self):
        return len(self.code)


class Compiler:
    def __init__(self):
        self.chunk = Chunk()

    def compile_program(self, program):
        for stmt in program.statements:
            self.compile_stmt(self.chunk, stmt)
        self.chunk.emit("HALT")
        return self.chunk.code

    # --- statements --------------------------------------------------------
    def compile_stmt(self, chunk, node):
        if isinstance(node, P.LetStmt):
            self.compile_expr(chunk, node.value)
            chunk.emit("DEFINE", node.name)
        elif isinstance(node, P.Assign):
            # Assignment used as a statement still flows through expr path.
            self.compile_expr(chunk, node)
            chunk.emit("POP")
        elif isinstance(node, P.PrintStmt):
            self.compile_expr(chunk, node.value)
            chunk.emit("PRINT")
        elif isinstance(node, P.IfStmt):
            self.compile_if(chunk, node)
        elif isinstance(node, P.WhileStmt):
            self.compile_while(chunk, node)
        elif isinstance(node, P.FnDecl):
            self.compile_fn(chunk, node)
        elif isinstance(node, P.ReturnStmt):
            if node.value is None:
                chunk.emit("CONST", None)
            else:
                self.compile_expr(chunk, node.value)
            chunk.emit("RETURN")
        elif isinstance(node, P.ExprStmt):
            self.compile_expr(chunk, node.expr)
            chunk.emit("POP")
        else:
            raise CompileError("unknown statement node: %r" % type(node).__name__)

    def compile_if(self, chunk, node):
        self.compile_expr(chunk, node.condition)
        else_jump = chunk.emit_jump("JUMP_IF_FALSE")
        for stmt in node.then_block:
            self.compile_stmt(chunk, stmt)
        if node.else_block is not None:
            end_jump = chunk.emit_jump("JUMP")
            chunk.patch_jump(else_jump)
            for stmt in node.else_block:
                self.compile_stmt(chunk, stmt)
            chunk.patch_jump(end_jump)
        else:
            chunk.patch_jump(else_jump)

    def compile_while(self, chunk, node):
        loop_start = chunk.here()
        self.compile_expr(chunk, node.condition)
        exit_jump = chunk.emit_jump("JUMP_IF_FALSE")
        for stmt in node.body:
            self.compile_stmt(chunk, stmt)
        back = chunk.emit_jump("JUMP")
        chunk.code[back] = ("JUMP", loop_start)
        chunk.patch_jump(exit_jump)

    def compile_fn(self, chunk, node):
        body_chunk = Chunk()
        for stmt in node.body:
            self.compile_stmt(body_chunk, stmt)
        # Implicit `return nil` if control falls off the end.
        body_chunk.emit("CONST", None)
        body_chunk.emit("RETURN")
        fn = Function(node.name, node.params, body_chunk.code)
        chunk.emit("CONST", fn)
        chunk.emit("DEFINE", node.name)

    # --- expressions -------------------------------------------------------
    def compile_expr(self, chunk, node):
        if isinstance(node, P.Literal):
            chunk.emit("CONST", node.value)
        elif isinstance(node, P.Variable):
            chunk.emit("LOAD", node.name)
        elif isinstance(node, P.Assign):
            self.compile_expr(chunk, node.value)
            # STORE leaves the assigned value on the stack so assignment is
            # itself an expression yielding that value.
            chunk.emit("STORE", node.name)
        elif isinstance(node, P.Unary):
            self.compile_expr(chunk, node.operand)
            chunk.emit("NEG" if node.op == "MINUS" else "NOT")
        elif isinstance(node, P.Binary):
            self.compile_expr(chunk, node.left)
            self.compile_expr(chunk, node.right)
            chunk.emit(BINARY_OP[node.op])
        elif isinstance(node, P.Logical):
            self.compile_logical(chunk, node)
        elif isinstance(node, P.Call):
            self.compile_call(chunk, node)
        else:
            raise CompileError("unknown expression node: %r" % type(node).__name__)

    def compile_logical(self, chunk, node):
        # Short-circuit. Leave the deciding operand value on the stack.
        self.compile_expr(chunk, node.left)
        chunk.emit("DUP")
        if node.op == "AND":
            # if left is falsey -> result is left; skip right.
            short = chunk.emit_jump("JUMP_IF_FALSE")
        else:  # OR
            short = chunk.emit_jump("JUMP_IF_TRUE")
        chunk.emit("POP")  # discard left, evaluate right
        self.compile_expr(chunk, node.right)
        chunk.patch_jump(short)

    def compile_call(self, chunk, node):
        # Evaluate callee, then args, then CALL <argc>.
        self.compile_expr(chunk, node.callee)
        for arg in node.args:
            self.compile_expr(chunk, arg)
        chunk.emit("CALL", len(node.args))


def compile_source(source):
    """Tokenize, parse, and compile a source string into a bytecode chunk."""
    program = P.parse(source)
    return Compiler().compile_program(program)


def compile_ast(program):
    return Compiler().compile_program(program)
