"""Compiler: compile AST to bytecode instructions."""

from enum import Enum, auto
from dataclasses import dataclass, field
from src.parser import (
    Program, NumberLit, StringLit, BoolLit, NilLit,
    Identifier, UnaryOp, BinaryOp, Assignment, CallExpr,
    LetStmt, PrintStmt, ExprStmt, Block, IfStmt, WhileStmt,
    FnDecl, ReturnStmt,
)


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
    op: OpCode
    arg: object = None


@dataclass
class CompiledFunction:
    name: str
    params: list[str]
    code: list[Instruction] = field(default_factory=list)
    constants: list[object] = field(default_factory=list)


@dataclass
class CompiledProgram:
    code: list[Instruction] = field(default_factory=list)
    constants: list[object] = field(default_factory=list)
    functions: dict[str, CompiledFunction] = field(default_factory=dict)


class Compiler:
    def __init__(self):
        self.code: list[Instruction] = []
        self.constants: list[object] = []
        self.functions: dict[str, CompiledFunction] = {}

    def emit(self, op: OpCode, arg=None) -> int:
        idx = len(self.code)
        self.code.append(Instruction(op, arg))
        return idx

    def add_constant(self, value: object) -> int:
        self.constants.append(value)
        return len(self.constants) - 1

    def compile_program(self, program: Program) -> CompiledProgram:
        for stmt in program.stmts:
            self.compile_stmt(stmt)
        self.emit(OpCode.HALT)
        return CompiledProgram(
            code=self.code,
            constants=self.constants,
            functions=self.functions,
        )

    def compile_stmt(self, node):
        if isinstance(node, LetStmt):
            self.compile_expr(node.value)
            self.emit(OpCode.STORE, node.name)
        elif isinstance(node, PrintStmt):
            self.compile_expr(node.value)
            self.emit(OpCode.PRINT)
        elif isinstance(node, ExprStmt):
            self.compile_expr(node.expr)
            self.emit(OpCode.POP)
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
                idx = self.add_constant(None)
                self.emit(OpCode.CONST, idx)
            self.emit(OpCode.RETURN)
        elif isinstance(node, Block):
            for stmt in node.stmts:
                self.compile_stmt(stmt)
        else:
            raise ValueError(f"Unknown statement type: {type(node)}")

    def compile_if(self, node: IfStmt):
        self.compile_expr(node.condition)
        jump_false = self.emit(OpCode.JUMP_IF_FALSE, None)

        for stmt in node.then_block.stmts:
            self.compile_stmt(stmt)

        if node.else_block is not None:
            jump_end = self.emit(OpCode.JUMP, None)
            self.code[jump_false].arg = len(self.code)
            for stmt in node.else_block.stmts:
                self.compile_stmt(stmt)
            self.code[jump_end].arg = len(self.code)
        else:
            self.code[jump_false].arg = len(self.code)

    def compile_while(self, node: WhileStmt):
        loop_start = len(self.code)
        self.compile_expr(node.condition)
        jump_false = self.emit(OpCode.JUMP_IF_FALSE, None)

        for stmt in node.body.stmts:
            self.compile_stmt(stmt)

        self.emit(OpCode.JUMP, loop_start)
        self.code[jump_false].arg = len(self.code)

    def compile_fn_decl(self, node: FnDecl):
        fn_compiler = Compiler()
        for stmt in node.body.stmts:
            fn_compiler.compile_stmt(stmt)
        # Implicit return nil if function doesn't end with return
        idx = fn_compiler.add_constant(None)
        fn_compiler.emit(OpCode.CONST, idx)
        fn_compiler.emit(OpCode.RETURN)

        self.functions[node.name] = CompiledFunction(
            name=node.name,
            params=node.params,
            code=fn_compiler.code,
            constants=fn_compiler.constants,
        )
        # Also propagate nested functions
        for fname, ffn in fn_compiler.functions.items():
            self.functions[fname] = ffn

    def compile_expr(self, node):
        if isinstance(node, NumberLit):
            idx = self.add_constant(node.value)
            self.emit(OpCode.CONST, idx)
        elif isinstance(node, StringLit):
            idx = self.add_constant(node.value)
            self.emit(OpCode.CONST, idx)
        elif isinstance(node, BoolLit):
            idx = self.add_constant(node.value)
            self.emit(OpCode.CONST, idx)
        elif isinstance(node, NilLit):
            idx = self.add_constant(None)
            self.emit(OpCode.CONST, idx)
        elif isinstance(node, Identifier):
            self.emit(OpCode.LOAD, node.name)
        elif isinstance(node, Assignment):
            self.compile_expr(node.value)
            self.emit(OpCode.STORE, node.name)
            self.emit(OpCode.LOAD, node.name)
        elif isinstance(node, UnaryOp):
            self.compile_expr(node.operand)
            if node.op == "-":
                self.emit(OpCode.NEG)
            elif node.op == "not":
                self.emit(OpCode.NOT)
        elif isinstance(node, BinaryOp):
            self.compile_binary(node)
        elif isinstance(node, CallExpr):
            for arg in node.args:
                self.compile_expr(arg)
            self.compile_expr(node.callee)
            self.emit(OpCode.CALL, len(node.args))
        else:
            raise ValueError(f"Unknown expression type: {type(node)}")

    def compile_binary(self, node: BinaryOp):
        op_map = {
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
            "and": OpCode.AND,
            "or": OpCode.OR,
        }
        self.compile_expr(node.left)
        self.compile_expr(node.right)
        self.emit(op_map[node.op])


def compile_ast(program) -> CompiledProgram:
    compiler = Compiler()
    return compiler.compile_program(program)
