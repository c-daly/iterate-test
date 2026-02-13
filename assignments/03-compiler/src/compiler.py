from enum import Enum, auto
from dataclasses import dataclass
from typing import Any

from parser import (
    NumberLit, StringLit, BoolLit, NilLit, Identifier,
    UnaryOp, BinOp, Assignment, LetStmt, IfStmt, WhileStmt,
    PrintStmt, FnDecl, ReturnStmt, CallExpr, ExprStmt,
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
    arg: Any = None


class CompileError(Exception):
    pass


BINOP_MAP = {
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


class Compiler:
    def compile(self, ast: list) -> list:
        code = []
        for node in ast:
            self._compile_node(node, code)
        code.append(Instruction(OpCode.HALT))
        return code

    def _compile_node(self, node, code: list):
        if isinstance(node, NumberLit):
            code.append(Instruction(OpCode.CONST, node.value))
        elif isinstance(node, StringLit):
            code.append(Instruction(OpCode.CONST, node.value))
        elif isinstance(node, BoolLit):
            code.append(Instruction(OpCode.CONST, node.value))
        elif isinstance(node, NilLit):
            code.append(Instruction(OpCode.CONST, None))
        elif isinstance(node, Identifier):
            code.append(Instruction(OpCode.LOAD, node.name))
        elif isinstance(node, UnaryOp):
            self._compile_node(node.operand, code)
            if node.op == "-":
                code.append(Instruction(OpCode.NEG))
            elif node.op == "not":
                code.append(Instruction(OpCode.NOT))
            else:
                raise CompileError(f"Unknown unary op: {node.op}")
        elif isinstance(node, BinOp):
            self._compile_node(node.left, code)
            self._compile_node(node.right, code)
            if node.op not in BINOP_MAP:
                raise CompileError(f"Unknown binary op: {node.op}")
            code.append(Instruction(BINOP_MAP[node.op]))
        elif isinstance(node, Assignment):
            self._compile_node(node.value, code)
            code.append(Instruction(OpCode.STORE, node.name))
            code.append(Instruction(OpCode.LOAD, node.name))
        elif isinstance(node, LetStmt):
            self._compile_node(node.value, code)
            code.append(Instruction(OpCode.STORE, node.name))
        elif isinstance(node, PrintStmt):
            self._compile_node(node.value, code)
            code.append(Instruction(OpCode.PRINT))
        elif isinstance(node, ExprStmt):
            self._compile_node(node.expr, code)
            code.append(Instruction(OpCode.POP))
        elif isinstance(node, IfStmt):
            self._compile_if(node, code)
        elif isinstance(node, WhileStmt):
            self._compile_while(node, code)
        elif isinstance(node, FnDecl):
            self._compile_fn_decl(node, code)
        elif isinstance(node, CallExpr):
            self._compile_call(node, code)
        elif isinstance(node, ReturnStmt):
            if node.value is not None:
                self._compile_node(node.value, code)
            else:
                code.append(Instruction(OpCode.CONST, None))
            code.append(Instruction(OpCode.RETURN))
        else:
            raise CompileError(f"Unknown AST node: {type(node).__name__}")

    def _compile_if(self, node: IfStmt, code: list):
        # Compile condition
        self._compile_node(node.condition, code)
        # JUMP_IF_FALSE placeholder
        jump_if_false_idx = len(code)
        code.append(Instruction(OpCode.JUMP_IF_FALSE, 0))
        # Compile then body
        for stmt in node.then_body:
            self._compile_node(stmt, code)
        if node.else_body is not None:
            # JUMP over else placeholder
            jump_over_else_idx = len(code)
            code.append(Instruction(OpCode.JUMP, 0))
            # Patch JUMP_IF_FALSE to here (start of else)
            code[jump_if_false_idx].arg = len(code)
            # Compile else body
            for stmt in node.else_body:
                self._compile_node(stmt, code)
            # Patch JUMP to here (after else)
            code[jump_over_else_idx].arg = len(code)
        else:
            # No else: patch JUMP_IF_FALSE to here
            code[jump_if_false_idx].arg = len(code)

    def _compile_while(self, node: WhileStmt, code: list):
        loop_start = len(code)
        # Compile condition
        self._compile_node(node.condition, code)
        # JUMP_IF_FALSE placeholder
        jump_if_false_idx = len(code)
        code.append(Instruction(OpCode.JUMP_IF_FALSE, 0))
        # Compile body
        for stmt in node.body:
            self._compile_node(stmt, code)
        # JUMP back to loop start
        code.append(Instruction(OpCode.JUMP, loop_start))
        # Patch JUMP_IF_FALSE to here (after loop)
        code[jump_if_false_idx].arg = len(code)

    def _compile_fn_decl(self, node: FnDecl, code: list):
        # Compile function body into separate instruction list
        fn_code = []
        for stmt in node.body:
            self._compile_node(stmt, fn_code)
        # Implicit return None at end of function
        fn_code.append(Instruction(OpCode.CONST, None))
        fn_code.append(Instruction(OpCode.RETURN))
        # Store as Function constant
        from vm import Function
        func = Function(node.name, node.params, fn_code)
        code.append(Instruction(OpCode.CONST, func))
        code.append(Instruction(OpCode.STORE, node.name))

    def _compile_call(self, node: CallExpr, code: list):
        # Compile args left-to-right
        for arg in node.args:
            self._compile_node(arg, code)
        # Compile callee
        self._compile_node(node.callee, code)
        # CALL with nargs
        code.append(Instruction(OpCode.CALL, len(node.args)))
