"""Compiler: lower AST to a flat list of bytecode instructions.

Produces a Chunk containing instructions for the top-level program plus
compiled function chunks referenced via constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Tuple

from .parser import (
    Assign,
    Binary,
    BoolLit,
    Call,
    ExprStmt,
    FnDecl,
    Identifier,
    IfStmt,
    LetStmt,
    Logical,
    NilLit,
    Node,
    NumberLit,
    PrintStmt,
    Program,
    ReturnStmt,
    StringLit,
    Unary,
    WhileStmt,
)


class CompileError(Exception):
    """Raised on a static compilation error."""


class OpCode(Enum):
    CONST = auto()
    POP = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    NOT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    LOAD = auto()
    STORE = auto()
    DEFINE = auto()
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    CALL = auto()
    RETURN = auto()
    PRINT = auto()
    MAKE_FN = auto()
    HALT = auto()


Instruction = Tuple[OpCode, Any]


@dataclass
class Function:
    """A compiled function: name, parameter list, body chunk, def-time env.

    `def_env` is the lexical environment that was active when the `fn`
    declaration ran. Set at runtime by the `MAKE_FN` opcode. Functions
    stored in the constant pool are templates with `def_env=None`; the
    VM produces a fresh, env-bound copy each time the declaration executes.
    """

    name: str
    params: List[str]
    chunk: "Chunk"
    def_env: Any = None  # Optional["Environment"]; typed loosely to avoid import cycle


@dataclass
class Chunk:
    """A flat list of instructions plus a constant pool."""

    code: List[Instruction] = field(default_factory=list)
    constants: List[Any] = field(default_factory=list)
    # Cache for O(1) primitive-constant deduping. Keyed by (type, value)
    # because Python conflates 1 == 1.0 == True under plain hashing/equality
    # but we treat them as distinct constants. Non-hashable values (Function,
    # list, ...) are simply appended without dedupe.
    _const_index: Dict[Tuple[type, Any], int] = field(default_factory=dict, repr=False)

    def add_const(self, value: Any) -> int:
        if isinstance(value, (int, float, str, bool)) or value is None:
            key = (type(value), value)
            cached = self._const_index.get(key)
            if cached is not None:
                return cached
            self.constants.append(value)
            idx = len(self.constants) - 1
            self._const_index[key] = idx
            return idx
        # Fall through for non-hashable / non-primitive values: append, no dedupe.
        self.constants.append(value)
        return len(self.constants) - 1

    def emit(self, op: OpCode, arg: Any = None) -> int:
        self.code.append((op, arg))
        return len(self.code) - 1

    def patch_jump(self, instr_index: int, target: int) -> None:
        op, _ = self.code[instr_index]
        self.code[instr_index] = (op, target)


_BIN_OPS = {
    "+": OpCode.ADD,
    "-": OpCode.SUB,
    "*": OpCode.MUL,
    "/": OpCode.DIV,
    "%": OpCode.MOD,
    "==": OpCode.EQ,
    "!=": OpCode.NE,
    "<": OpCode.LT,
    ">": OpCode.GT,
    "<=": OpCode.LE,
    ">=": OpCode.GE,
}


class Compiler:
    """Walk the AST and emit bytecode."""

    def __init__(self) -> None:
        self.chunk = Chunk()

    def compile(self, program: Program) -> Chunk:
        for stmt in program.statements:
            self._compile_stmt(self.chunk, stmt)
        self.chunk.emit(OpCode.HALT)
        return self.chunk

    # ----------------------------------------------------------- statements
    def _compile_stmt(self, chunk: Chunk, node: Node) -> None:
        if isinstance(node, LetStmt):
            self._compile_expr(chunk, node.value)
            chunk.emit(OpCode.DEFINE, node.name)
            return
        if isinstance(node, IfStmt):
            self._compile_if(chunk, node)
            return
        if isinstance(node, WhileStmt):
            self._compile_while(chunk, node)
            return
        if isinstance(node, PrintStmt):
            self._compile_expr(chunk, node.expr)
            chunk.emit(OpCode.PRINT)
            return
        if isinstance(node, FnDecl):
            self._compile_fn(chunk, node)
            return
        if isinstance(node, ReturnStmt):
            if node.expr is None:
                idx = chunk.add_const(None)
                chunk.emit(OpCode.CONST, idx)
            else:
                self._compile_expr(chunk, node.expr)
            chunk.emit(OpCode.RETURN)
            return
        if isinstance(node, ExprStmt):
            self._compile_expr(chunk, node.expr)
            chunk.emit(OpCode.POP)
            return
        raise CompileError(f"Unknown statement: {type(node).__name__}")

    def _compile_if(self, chunk: Chunk, node: IfStmt) -> None:
        self._compile_expr(chunk, node.cond)
        jmp_else = chunk.emit(OpCode.JUMP_IF_FALSE, -1)
        chunk.emit(OpCode.POP)
        for stmt in node.then_block:
            self._compile_stmt(chunk, stmt)
        jmp_end = chunk.emit(OpCode.JUMP, -1)
        chunk.patch_jump(jmp_else, len(chunk.code))
        chunk.emit(OpCode.POP)
        if node.else_block is not None:
            for stmt in node.else_block:
                self._compile_stmt(chunk, stmt)
        chunk.patch_jump(jmp_end, len(chunk.code))

    def _compile_while(self, chunk: Chunk, node: WhileStmt) -> None:
        loop_start = len(chunk.code)
        self._compile_expr(chunk, node.cond)
        jmp_end = chunk.emit(OpCode.JUMP_IF_FALSE, -1)
        chunk.emit(OpCode.POP)
        for stmt in node.body:
            self._compile_stmt(chunk, stmt)
        chunk.emit(OpCode.JUMP, loop_start)
        chunk.patch_jump(jmp_end, len(chunk.code))
        chunk.emit(OpCode.POP)

    def _compile_fn(self, chunk: Chunk, node: FnDecl) -> None:
        body_chunk = Chunk()
        sub = Compiler()
        sub.chunk = body_chunk
        for stmt in node.body:
            sub._compile_stmt(body_chunk, stmt)
        # implicit return nil if no explicit return at end
        idx = body_chunk.add_const(None)
        body_chunk.emit(OpCode.CONST, idx)
        body_chunk.emit(OpCode.RETURN)
        # Build a *template* Function (def_env unset). MAKE_FN at runtime
        # produces a fresh, env-bound copy each time the declaration runs,
        # giving lexical (static) scoping for free variables.
        fn_template = Function(name=node.name, params=list(node.params), chunk=body_chunk)
        const_idx = chunk.add_const(fn_template)
        chunk.emit(OpCode.CONST, const_idx)
        chunk.emit(OpCode.MAKE_FN)
        chunk.emit(OpCode.DEFINE, node.name)

    # ----------------------------------------------------------- expressions
    def _compile_expr(self, chunk: Chunk, node: Node) -> None:
        if isinstance(node, NumberLit):
            idx = chunk.add_const(node.value)
            chunk.emit(OpCode.CONST, idx)
            return
        if isinstance(node, StringLit):
            idx = chunk.add_const(node.value)
            chunk.emit(OpCode.CONST, idx)
            return
        if isinstance(node, BoolLit):
            idx = chunk.add_const(node.value)
            chunk.emit(OpCode.CONST, idx)
            return
        if isinstance(node, NilLit):
            idx = chunk.add_const(None)
            chunk.emit(OpCode.CONST, idx)
            return
        if isinstance(node, Identifier):
            chunk.emit(OpCode.LOAD, node.name)
            return
        if isinstance(node, Assign):
            self._compile_expr(chunk, node.value)
            chunk.emit(OpCode.STORE, node.name)
            return
        if isinstance(node, Unary):
            self._compile_expr(chunk, node.operand)
            if node.op == "-":
                chunk.emit(OpCode.NEG)
            elif node.op == "not":
                chunk.emit(OpCode.NOT)
            else:
                raise CompileError(f"Unknown unary op: {node.op}")
            return
        if isinstance(node, Binary):
            self._compile_expr(chunk, node.left)
            self._compile_expr(chunk, node.right)
            if node.op not in _BIN_OPS:
                raise CompileError(f"Unknown binary op: {node.op}")
            chunk.emit(_BIN_OPS[node.op])
            return
        if isinstance(node, Logical):
            self._compile_logical(chunk, node)
            return
        if isinstance(node, Call):
            for arg in node.args:
                self._compile_expr(chunk, arg)
            self._compile_expr(chunk, node.callee)
            chunk.emit(OpCode.CALL, len(node.args))
            return
        raise CompileError(f"Unknown expression: {type(node).__name__}")

    def _compile_logical(self, chunk: Chunk, node: Logical) -> None:
        # Short-circuit lowering. Leave a single boolean-equivalent value
        # on top of the stack.
        if node.op == "and":
            # left; if false skip right and use left as result
            self._compile_expr(chunk, node.left)
            jmp_short = chunk.emit(OpCode.JUMP_IF_FALSE, -1)
            chunk.emit(OpCode.POP)
            self._compile_expr(chunk, node.right)
            chunk.patch_jump(jmp_short, len(chunk.code))
            return
        if node.op == "or":
            # left; if true skip right and use left as result
            self._compile_expr(chunk, node.left)
            jmp_to_right = chunk.emit(OpCode.JUMP_IF_FALSE, -1)
            jmp_end = chunk.emit(OpCode.JUMP, -1)
            chunk.patch_jump(jmp_to_right, len(chunk.code))
            chunk.emit(OpCode.POP)
            self._compile_expr(chunk, node.right)
            chunk.patch_jump(jmp_end, len(chunk.code))
            return
        raise CompileError(f"Unknown logical op: {node.op}")
