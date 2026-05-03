"""AST -> bytecode compiler.

Bytecode is a list of (op, *args) tuples. Function bodies are themselves
compiled into a CodeObject (own bytecode + param list + locals table) and
referenced via a CONST instruction that pushes the CodeObject onto the stack.

Scoping
-------
Variables bind to either:
  * a local frame slot (function locals + params), or
  * the enclosing function's scope chain (looked up at runtime via LOAD/STORE
    falling back to parent frames).

The compiler does not do static slot allocation; all variables are name-keyed
at runtime. STORE writes to the nearest existing binding, or creates a new
binding in the current scope if none exists. LOAD walks the scope chain.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

from . import parser as P


@dataclass
class CodeObject:
    name: str
    params: List[str] = field(default_factory=list)
    instructions: List[tuple] = field(default_factory=list)

    def __repr__(self) -> str:
        return "<CodeObject " + self.name + " params=" + str(self.params) + " len=" + str(len(self.instructions)) + ">"


class CompileError(Exception):
    pass


class Compiler:
    def __init__(self, name: str = "<main>", params: Optional[List[str]] = None) -> None:
        self.code = CodeObject(name=name, params=list(params or []))

    # ---- emission helpers -----------------------------------------
    def _emit(self, op: str, *args: Any) -> int:
        idx = len(self.code.instructions)
        self.code.instructions.append((op, *args))
        return idx

    def _patch(self, idx: int, *new_args: Any) -> None:
        op = self.code.instructions[idx][0]
        self.code.instructions[idx] = (op, *new_args)

    # ---- entry points ---------------------------------------------
    def compile_program(self, program: P.Program) -> CodeObject:
        for stmt in program.statements:
            self._stmt(stmt)
        self._emit("HALT")
        return self.code

    def compile_function(self, fn: P.FnDecl) -> CodeObject:
        for stmt in fn.body:
            self._stmt(stmt)
        # implicit return nil
        self._emit("CONST", None)
        self._emit("RETURN")
        return self.code

    # ---- statements -----------------------------------------------
    def _stmt(self, node: Any) -> None:
        if isinstance(node, P.Let):
            self._expr(node.value)
            # let always creates a local binding
            self._emit("STORE", node.name, True)  # True = declare local
            return
        if isinstance(node, P.Print):
            self._expr(node.value)
            self._emit("PRINT")
            return
        if isinstance(node, P.ExprStmt):
            self._expr(node.expr)
            self._emit("POP")
            return
        if isinstance(node, P.If):
            self._expr(node.cond)
            jf = self._emit("JUMP_IF_FALSE", -1)
            for s in node.then_block:
                self._stmt(s)
            if node.else_block is not None:
                jend = self._emit("JUMP", -1)
                self._patch(jf, len(self.code.instructions))
                for s in node.else_block:
                    self._stmt(s)
                self._patch(jend, len(self.code.instructions))
            else:
                self._patch(jf, len(self.code.instructions))
            return
        if isinstance(node, P.While):
            loop_start = len(self.code.instructions)
            self._expr(node.cond)
            jf = self._emit("JUMP_IF_FALSE", -1)
            for s in node.body:
                self._stmt(s)
            self._emit("JUMP", loop_start)
            self._patch(jf, len(self.code.instructions))
            return
        if isinstance(node, P.FnDecl):
            sub = Compiler(name=node.name, params=node.params)
            code_obj = sub.compile_function(node)
            self._emit("CONST", code_obj)
            self._emit("STORE", node.name, True)
            return
        if isinstance(node, P.Return):
            if node.value is None:
                self._emit("CONST", None)
            else:
                self._expr(node.value)
            self._emit("RETURN")
            return
        raise CompileError("Unknown statement node: " + type(node).__name__)

    # ---- expressions ----------------------------------------------
    def _expr(self, node: Any) -> None:
        if isinstance(node, P.Number):
            self._emit("CONST", node.value)
            return
        if isinstance(node, P.String):
            self._emit("CONST", node.value)
            return
        if isinstance(node, P.Bool):
            self._emit("CONST", node.value)
            return
        if isinstance(node, P.Nil):
            self._emit("CONST", None)
            return
        if isinstance(node, P.Ident):
            self._emit("LOAD", node.name)
            return
        if isinstance(node, P.Assign):
            self._expr(node.value)
            # duplicate so assignment is also an expression value
            self._emit("STORE", node.name, False)  # False = assign existing
            self._emit("LOAD", node.name)
            return
        if isinstance(node, P.Binary):
            self._expr(node.left)
            self._expr(node.right)
            op_map = {
                "+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "%": "MOD",
                "==": "EQ", "!=": "NE", "<": "LT", ">": "GT", "<=": "LE", ">=": "GE",
            }
            self._emit(op_map[node.op])
            return
        if isinstance(node, P.Unary):
            self._expr(node.operand)
            if node.op == "-":
                self._emit("NEG")
            else:
                self._emit("NOT")
            return
        if isinstance(node, P.Logical):
            # Short-circuit using JUMP_IF_FALSE / JUMP
            self._expr(node.left)
            if node.op == "and":
                # if left falsy, leave it (the falsy value) and jump past right
                jmp = self._emit("JUMP_IF_FALSE_KEEP", -1)
                self._emit("POP")
                self._expr(node.right)
                self._patch(jmp, len(self.code.instructions))
            else:  # or
                jmp = self._emit("JUMP_IF_TRUE_KEEP", -1)
                self._emit("POP")
                self._expr(node.right)
                self._patch(jmp, len(self.code.instructions))
            return
        if isinstance(node, P.Call):
            self._expr(node.callee)
            for a in node.args:
                self._expr(a)
            self._emit("CALL", len(node.args))
            return
        raise CompileError("Unknown expression node: " + type(node).__name__)


def compile_program(program: P.Program) -> CodeObject:
    return Compiler(name="<main>").compile_program(program)
