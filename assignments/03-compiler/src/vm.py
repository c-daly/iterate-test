"""Stack-based virtual machine that runs compiled Chunks.

The VM maintains a value stack, a call frame stack, and a chain of
environments to support lexical lookup of outer variables from inside
functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from .compiler import Chunk, Function, OpCode


class RuntimeError_(Exception):
    """Raised on a runtime error inside the VM."""


@dataclass
class Environment:
    """A lexical environment with optional parent for outer-variable lookup."""

    values: Dict[str, Any] = field(default_factory=dict)
    parent: Optional["Environment"] = None

    def define(self, name: str, value: Any) -> None:
        self.values[name] = value

    def get(self, name: str) -> Any:
        env: Optional[Environment] = self
        while env is not None:
            if name in env.values:
                return env.values[name]
            env = env.parent
        raise RuntimeError_(f"Undefined variable: {name}")

    def set(self, name: str, value: Any) -> None:
        env: Optional[Environment] = self
        while env is not None:
            if name in env.values:
                env.values[name] = value
                return
            env = env.parent
        raise RuntimeError_(f"Cannot assign to undefined variable: {name}")


@dataclass
class Frame:
    """One call frame: chunk, instruction pointer, environment."""

    chunk: Chunk
    ip: int = 0
    env: Environment = field(default_factory=Environment)


TRUTHY_FALSE = (False, None, 0, 0.0, "")


def _is_truthy(value: Any) -> bool:
    return value not in TRUTHY_FALSE


def _format(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "nil"
    return str(value)


class VM:
    """Stack-based bytecode VM."""

    def __init__(self) -> None:
        self.stack: List[Any] = []
        self.frames: List[Frame] = []
        self.output: List[str] = []

    def run(self, chunk: Chunk) -> List[str]:
        global_env = Environment()
        self.frames.append(Frame(chunk=chunk, ip=0, env=global_env))
        while self.frames:
            frame = self.frames[-1]
            if frame.ip >= len(frame.chunk.code):
                # implicit halt for top-level only; functions emit RETURN
                self.frames.pop()
                continue
            op, arg = frame.chunk.code[frame.ip]
            frame.ip += 1
            if not self._dispatch(op, arg, frame):
                break
        return self.output

    # ----------------------------------------------------------- dispatch
    def _dispatch(self, op: OpCode, arg: Any, frame: Frame) -> bool:
        if op == OpCode.CONST:
            self.stack.append(frame.chunk.constants[arg])
            return True
        if op == OpCode.POP:
            self.stack.pop()
            return True
        if op == OpCode.ADD:
            self._binop_add()
            return True
        if op == OpCode.SUB:
            self._numeric(lambda a, b: a - b, "-")
            return True
        if op == OpCode.MUL:
            self._numeric(lambda a, b: a * b, "*")
            return True
        if op == OpCode.DIV:
            self._div()
            return True
        if op == OpCode.MOD:
            self._mod()
            return True
        if op == OpCode.NEG:
            v = self.stack.pop()
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise RuntimeError_(f"Cannot negate {type(v).__name__}")
            self.stack.append(-v)
            return True
        if op == OpCode.NOT:
            v = self.stack.pop()
            self.stack.append(not _is_truthy(v))
            return True
        if op == OpCode.EQ:
            b = self.stack.pop()
            a = self.stack.pop()
            self.stack.append(a == b)
            return True
        if op == OpCode.NE:
            b = self.stack.pop()
            a = self.stack.pop()
            self.stack.append(a != b)
            return True
        if op == OpCode.LT:
            self._cmp(lambda a, b: a < b, "<")
            return True
        if op == OpCode.GT:
            self._cmp(lambda a, b: a > b, ">")
            return True
        if op == OpCode.LE:
            self._cmp(lambda a, b: a <= b, "<=")
            return True
        if op == OpCode.GE:
            self._cmp(lambda a, b: a >= b, ">=")
            return True
        if op == OpCode.LOAD:
            self.stack.append(frame.env.get(arg))
            return True
        if op == OpCode.STORE:
            value = self.stack[-1]
            frame.env.set(arg, value)
            return True
        if op == OpCode.DEFINE:
            value = self.stack.pop()
            frame.env.define(arg, value)
            return True
        if op == OpCode.JUMP:
            frame.ip = arg
            return True
        if op == OpCode.JUMP_IF_FALSE:
            cond = self.stack[-1]
            if not _is_truthy(cond):
                frame.ip = arg
            return True
        if op == OpCode.MAKE_FN:
            template = self.stack.pop()
            if not isinstance(template, Function):
                raise RuntimeError_(f"MAKE_FN expected Function, got {type(template).__name__}")
            # Bind a fresh copy of the function to the env active at
            # def-time. This gives lexical (static) scoping: free variables
            # resolve against where the function was defined, not where it
            # is called.
            self.stack.append(replace(template, def_env=frame.env))
            return True
        if op == OpCode.CALL:
            self._call(arg, frame)
            return True
        if op == OpCode.RETURN:
            value = self.stack.pop()
            self.frames.pop()
            self.stack.append(value)
            return True
        if op == OpCode.PRINT:
            v = self.stack.pop()
            self.output.append(_format(v))
            return True
        if op == OpCode.HALT:
            return False
        raise RuntimeError_(f"Unknown opcode: {op}")

    # ----------------------------------------------------------- helpers
    def _binop_add(self) -> None:
        b = self.stack.pop()
        a = self.stack.pop()
        if isinstance(a, str) and isinstance(b, str):
            self.stack.append(a + b)
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            self.stack.append(a + b)
            return
        raise RuntimeError_(f"Type error: cannot add {type(a).__name__} and {type(b).__name__}")

    def _numeric(self, func, name: str) -> None:
        b = self.stack.pop()
        a = self.stack.pop()
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) \
                or isinstance(a, bool) or isinstance(b, bool):
            raise RuntimeError_(f"Type error: cannot apply {name} to {type(a).__name__} and {type(b).__name__}")
        self.stack.append(func(a, b))

    def _div(self) -> None:
        b = self.stack.pop()
        a = self.stack.pop()
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) \
                or isinstance(a, bool) or isinstance(b, bool):
            raise RuntimeError_(f"Type error: cannot divide {type(a).__name__} and {type(b).__name__}")
        if b == 0:
            raise RuntimeError_("Division by zero")
        if isinstance(a, int) and isinstance(b, int):
            # integer division for int / int when result is exact, else float
            if a % b == 0:
                self.stack.append(a // b)
                return
        self.stack.append(a / b)

    def _mod(self) -> None:
        b = self.stack.pop()
        a = self.stack.pop()
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) \
                or isinstance(a, bool) or isinstance(b, bool):
            raise RuntimeError_(f"Type error: cannot mod {type(a).__name__} and {type(b).__name__}")
        if b == 0:
            raise RuntimeError_("Modulo by zero")
        self.stack.append(a % b)

    def _cmp(self, func, name: str) -> None:
        b = self.stack.pop()
        a = self.stack.pop()
        try:
            self.stack.append(func(a, b))
        except TypeError as exc:
            raise RuntimeError_(f"Type error in {name}: {exc}")

    def _call(self, arg_count: int, caller: Frame) -> None:
        callee = self.stack.pop()
        if not isinstance(callee, Function):
            raise RuntimeError_(f"Cannot call non-function: {type(callee).__name__}")
        if len(callee.params) != arg_count:
            raise RuntimeError_(
                f"Function {callee.name} expected {len(callee.params)} args, got {arg_count}"
            )
        args = [self.stack.pop() for _ in range(arg_count)]
        args.reverse()
        # Lexical scoping: chain off the env active where the function was
        # *defined* (captured at MAKE_FN time), not where it is called.
        # Fall back to the caller env if def_env is missing (e.g. a Function
        # constructed by hand in tests rather than via MAKE_FN).
        parent_env = callee.def_env if callee.def_env is not None else caller.env
        new_env = Environment(parent=parent_env)
        for param, value in zip(callee.params, args):
            new_env.define(param, value)
        self.frames.append(Frame(chunk=callee.chunk, ip=0, env=new_env))
