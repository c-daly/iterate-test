"""Compiler - compiles AST to bytecode instructions."""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Any
from src.parser import (
    Program, LetStmt, PrintStmt, ExprStmt, IfStmt, WhileStmt, FnDecl,
    ReturnStmt, Block, NumberLit, StringLit, BoolLit, NilLit, Ident,
    UnaryOp, BinOp, Assign, Call,
)


class Op(Enum):
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
    AND = auto()
    OR = auto()
    LOAD = auto()
    STORE = auto()
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    CALL = auto()
    RETURN = auto()
    PRINT = auto()
    HALT = auto()


@dataclass
class Instruction:
    op: Op
    arg: Any = None

    def __repr__(self):
        if self.arg is not None:
            return f"{self.op.name} {self.arg!r}"
        return self.op.name


@dataclass
class CompiledFunction:
    name: str
    params: list
    code: list
    num_locals: int = 0


class CompileError(Exception):
    pass


class Compiler:
    def __init__(self):
        self.code: List[Instruction] = []
        self.constants: List[Any] = []
        self.functions: dict = {}  # name -> CompiledFunction

    def compile(self, program: Program) -> List[Instruction]:
        for stmt in program.stmts:
            self.compile_stmt(stmt)
        self.emit(Op.HALT)
        return self.code

    def emit(self, op: Op, arg=None) -> int:
        idx = len(self.code)
        self.code.append(Instruction(op, arg))
        return idx

    def compile_stmt(self, node):
        if isinstance(node, LetStmt):
            self.compile_expr(node.value)
            self.emit(Op.STORE, node.name)
        elif isinstance(node, PrintStmt):
            self.compile_expr(node.value)
            self.emit(Op.PRINT)
        elif isinstance(node, ExprStmt):
            self.compile_expr(node.expr)
            self.emit(Op.POP)
        elif isinstance(node, IfStmt):
            self.compile_if(node)
        elif isinstance(node, WhileStmt):
            self.compile_while(node)
        elif isinstance(node, FnDecl):
            self.compile_fn_decl(node)
        elif isinstance(node, ReturnStmt):
            if node.value is not None:
                self.compile_expr(node.value)
            else:
                self.emit(Op.CONST, None)
            self.emit(Op.RETURN)
        elif isinstance(node, Block):
            for s in node.stmts:
                self.compile_stmt(s)
        else:
            raise CompileError(f"Unknown statement type: {type(node).__name__}")

    def compile_if(self, node: IfStmt):
        self.compile_expr(node.condition)
        jump_false = self.emit(Op.JUMP_IF_FALSE, None)
        self.compile_stmt(node.then_block)
        if node.else_block is not None:
            jump_end = self.emit(Op.JUMP, None)
            self.code[jump_false].arg = len(self.code)
            self.compile_stmt(node.else_block)
            self.code[jump_end].arg = len(self.code)
        else:
            self.code[jump_false].arg = len(self.code)

    def compile_while(self, node: WhileStmt):
        loop_start = len(self.code)
        self.compile_expr(node.condition)
        jump_false = self.emit(Op.JUMP_IF_FALSE, None)
        self.compile_stmt(node.body)
        self.emit(Op.JUMP, loop_start)
        self.code[jump_false].arg = len(self.code)

    def compile_fn_decl(self, node: FnDecl):
        # Save current state
        outer_code = self.code
        self.code = []

        # Compile function body
        for stmt in node.body.stmts:
            self.compile_stmt(stmt)

        # Implicit return nil
        self.emit(Op.CONST, None)
        self.emit(Op.RETURN)

        fn = CompiledFunction(
            name=node.name,
            params=node.params,
            code=self.code,
        )

        # Restore outer code
        self.code = outer_code
        self.functions[node.name] = fn
        # Store function reference as a constant
        self.emit(Op.CONST, fn)
        self.emit(Op.STORE, node.name)

    def compile_expr(self, node):
        if isinstance(node, NumberLit):
            self.emit(Op.CONST, node.value)
        elif isinstance(node, StringLit):
            self.emit(Op.CONST, node.value)
        elif isinstance(node, BoolLit):
            self.emit(Op.CONST, node.value)
        elif isinstance(node, NilLit):
            self.emit(Op.CONST, None)
        elif isinstance(node, Ident):
            self.emit(Op.LOAD, node.name)
        elif isinstance(node, Assign):
            self.compile_expr(node.value)
            self.emit(Op.STORE, node.name)
            self.emit(Op.LOAD, node.name)  # assignment is an expression, push value back
        elif isinstance(node, UnaryOp):
            self.compile_expr(node.operand)
            if node.op == "-":
                self.emit(Op.NEG)
            elif node.op == "not":
                self.emit(Op.NOT)
        elif isinstance(node, BinOp):
            if node.op == "and":
                self.compile_and(node)
            elif node.op == "or":
                self.compile_or(node)
            else:
                self.compile_expr(node.left)
                self.compile_expr(node.right)
                op_map = {
                    "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.MOD,
                    "==": Op.EQ, "!=": Op.NE, "<": Op.LT, ">": Op.GT,
                    "<=": Op.LE, ">=": Op.GE,
                }
                self.emit(op_map[node.op])
        elif isinstance(node, Call):
            # Push arguments
            for arg in node.args:
                self.compile_expr(arg)
            # Push callee
            self.compile_expr(node.callee)
            self.emit(Op.CALL, len(node.args))
        else:
            raise CompileError(f"Unknown expression type: {type(node).__name__}")

    def compile_and(self, node: BinOp):
        self.compile_expr(node.left)
        jump_false = self.emit(Op.JUMP_IF_FALSE, None)
        self.compile_expr(node.right)
        jump_end = self.emit(Op.JUMP, None)
        self.code[jump_false].arg = len(self.code)
        self.emit(Op.CONST, False)
        self.code[jump_end].arg = len(self.code)

    def compile_or(self, node: BinOp):
        self.compile_expr(node.left)
        # If truthy, short-circuit
        jump_true = self.emit(Op.JUMP_IF_FALSE, None)
        self.emit(Op.CONST, True)
        jump_end = self.emit(Op.JUMP, None)
        self.code[jump_true].arg = len(self.code)
        self.compile_expr(node.right)
        self.code[jump_end].arg = len(self.code)
