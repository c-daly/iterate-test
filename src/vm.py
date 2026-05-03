"""Stack-based virtual machine.

Executes a CodeObject produced by compiler.py. Returns the list of printed
output strings. Variable scoping is implemented as a chain of frames: each
function call pushes a Frame whose locals dict is parented to the function's
defining environment (closure-style; we capture defining-environment when a
CodeObject value is loaded via CONST since that's when STORE binds the name).

Truthiness rules:
  * False, None, 0, 0.0, empty-string -> falsy
  * everything else truthy
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .compiler import CodeObject


class VMError(Exception):
    pass


@dataclass
class Frame:
    code: CodeObject
    ip: int = 0
    locals: Dict[str, Any] = field(default_factory=dict)
    parent: Optional["Frame"] = None
    return_to: Optional[int] = None
    return_frame: Optional["Frame"] = None
    base_stack_len: int = 0


@dataclass
class Closure:
    code: CodeObject
    env: Optional[Frame]


def _truthy(v: Any) -> bool:
    if v is False or v is None:
        return False
    if isinstance(v, (int, float)) and v == 0:
        return False
    if isinstance(v, str) and v == "":
        return False
    return True


def _lookup(frame: Frame, name: str) -> Any:
    cur: Optional[Frame] = frame
    while cur is not None:
        if name in cur.locals:
            return cur.locals[name]
        cur = cur.parent
    raise VMError("Undefined variable: " + name)


def _store_existing(frame: Frame, name: str, value: Any) -> bool:
    cur: Optional[Frame] = frame
    while cur is not None:
        if name in cur.locals:
            cur.locals[name] = value
            return True
        cur = cur.parent
    return False


def run(code: CodeObject) -> List[str]:
    output: List[str] = []
    stack: List[Any] = []
    main_frame = Frame(code=code, locals={}, parent=None)
    frame = main_frame

    while True:
        instr = frame.code.instructions[frame.ip]
        op = instr[0]
        args = instr[1:]
        frame.ip += 1

        if op == "CONST":
            v = args[0]
            if isinstance(v, CodeObject):
                stack.append(Closure(code=v, env=frame))
            else:
                stack.append(v)
        elif op == "POP":
            stack.pop()
        elif op == "ADD":
            b = stack.pop()
            a = stack.pop()
            if isinstance(a, str) or isinstance(b, str):
                if not (isinstance(a, str) and isinstance(b, str)):
                    raise VMError("Cannot add string to non-string")
                stack.append(a + b)
            else:
                stack.append(a + b)
        elif op == "SUB":
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif op == "MUL":
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif op == "DIV":
            b = stack.pop()
            a = stack.pop()
            if b == 0:
                raise VMError("Division by zero")
            if isinstance(a, float) or isinstance(b, float):
                stack.append(a / b)
            elif a % b == 0:
                stack.append(a // b)
            else:
                stack.append(a / b)
        elif op == "MOD":
            b = stack.pop()
            a = stack.pop()
            if b == 0:
                raise VMError("Modulo by zero")
            stack.append(a % b)
        elif op == "NEG":
            a = stack.pop()
            stack.append(-a)
        elif op == "NOT":
            a = stack.pop()
            stack.append(not _truthy(a))
        elif op == "EQ":
            b = stack.pop()
            a = stack.pop()
            stack.append(a == b)
        elif op == "NE":
            b = stack.pop()
            a = stack.pop()
            stack.append(a != b)
        elif op == "LT":
            b = stack.pop()
            a = stack.pop()
            stack.append(a < b)
        elif op == "GT":
            b = stack.pop()
            a = stack.pop()
            stack.append(a > b)
        elif op == "LE":
            b = stack.pop()
            a = stack.pop()
            stack.append(a <= b)
        elif op == "GE":
            b = stack.pop()
            a = stack.pop()
            stack.append(a >= b)
        elif op == "AND":
            b = stack.pop()
            a = stack.pop()
            stack.append(a if not _truthy(a) else b)
        elif op == "OR":
            b = stack.pop()
            a = stack.pop()
            stack.append(a if _truthy(a) else b)
        elif op == "LOAD":
            name = args[0]
            stack.append(_lookup(frame, name))
        elif op == "STORE":
            name = args[0]
            declare_local = args[1] if len(args) > 1 else False
            value = stack.pop()
            if declare_local:
                frame.locals[name] = value
            else:
                if not _store_existing(frame, name, value):
                    frame.locals[name] = value
        elif op == "JUMP":
            frame.ip = args[0]
        elif op == "JUMP_IF_FALSE":
            v = stack.pop()
            if not _truthy(v):
                frame.ip = args[0]
        elif op == "JUMP_IF_FALSE_KEEP":
            v = stack[-1]
            if not _truthy(v):
                frame.ip = args[0]
        elif op == "JUMP_IF_TRUE_KEEP":
            v = stack[-1]
            if _truthy(v):
                frame.ip = args[0]
        elif op == "CALL":
            argc = args[0]
            call_args = [stack.pop() for _ in range(argc)][::-1]
            callee = stack.pop()
            if not isinstance(callee, Closure):
                raise VMError("Cannot call non-function value: " + repr(callee))
            if len(call_args) != len(callee.code.params):
                raise VMError(
                    "Arity mismatch calling " + callee.code.name
                    + ": expected " + str(len(callee.code.params))
                    + " got " + str(len(call_args))
                )
            new_frame = Frame(
                code=callee.code,
                ip=0,
                locals={p: a for p, a in zip(callee.code.params, call_args)},
                parent=callee.env,
                return_to=frame.ip,
                return_frame=frame,
                base_stack_len=len(stack),
            )
            frame = new_frame
        elif op == "RETURN":
            ret = stack.pop()
            while len(stack) > frame.base_stack_len:
                stack.pop()
            caller = frame.return_frame
            if caller is None:
                return output
            ret_ip = frame.return_to
            frame = caller
            if ret_ip is not None:
                frame.ip = ret_ip
            stack.append(ret)
        elif op == "PRINT":
            v = stack.pop()
            output.append(_format(v))
        elif op == "HALT":
            return output
        else:
            raise VMError("Unknown opcode: " + str(op))


def _format(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "nil"
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v)
    return str(v)
