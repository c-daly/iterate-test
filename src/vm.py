"""Stack-based virtual machine for executing bytecode."""

from dataclasses import dataclass, field
from typing import Any, List, Dict
from src.compiler import Op, Instruction, CompiledFunction


class RuntimeError_(Exception):
    pass


@dataclass
class CallFrame:
    code: List[Instruction]
    ip: int
    locals: Dict[str, Any]
    return_to_ip: int
    return_to_code: List[Instruction]
    return_to_locals: Dict[str, Any]


class VM:
    def __init__(self, code: List[Instruction], functions: dict = None):
        self.code = code
        self.ip = 0
        self.stack: List[Any] = []
        self.globals: Dict[str, Any] = {}
        self.locals: Dict[str, Any] = {}
        self.call_stack: List[CallFrame] = []
        self.output: List[Any] = []
        self.functions = functions or {}

    def push(self, value: Any):
        self.stack.append(value)

    def pop(self) -> Any:
        if not self.stack:
            raise RuntimeError_("Stack underflow")
        return self.stack.pop()

    def run(self) -> List[Any]:
        while self.ip < len(self.code):
            instr = self.code[self.ip]
            self.ip += 1
            self.execute(instr)
        return self.output

    def is_truthy(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        return True

    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric but NOT a bool."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def execute(self, instr: Instruction):
        op = instr.op

        if op == Op.CONST:
            self.push(instr.arg)

        elif op == Op.POP:
            self.pop()

        elif op == Op.ADD:
            b, a = self.pop(), self.pop()
            if isinstance(a, str) or isinstance(b, str):
                self.push(str(a) if not isinstance(a, str) else a)
                b_str = str(b) if not isinstance(b, str) else b
                # Redo: proper string concat
                result = (str(a) if not isinstance(a, str) else a) + (str(b) if not isinstance(b, str) else b)
                self.stack.pop()  # remove the a we just pushed
                self.push(result)
            elif self._is_numeric(a) and self._is_numeric(b):
                self.push(a + b)
            else:
                raise RuntimeError_(f"Cannot add {type(a).__name__} and {type(b).__name__}")

        elif op == Op.SUB:
            b, a = self.pop(), self.pop()
            self._check_numeric(a, b, "-")
            self.push(a - b)

        elif op == Op.MUL:
            b, a = self.pop(), self.pop()
            self._check_numeric(a, b, "*")
            self.push(a * b)

        elif op == Op.DIV:
            b, a = self.pop(), self.pop()
            self._check_numeric(a, b, "/")
            if b == 0:
                raise RuntimeError_("Division by zero")
            if isinstance(a, int) and isinstance(b, int):
                result = a // b
                self.push(result)
            else:
                self.push(a / b)

        elif op == Op.MOD:
            b, a = self.pop(), self.pop()
            self._check_numeric(a, b, "%")
            if b == 0:
                raise RuntimeError_("Division by zero")
            self.push(a % b)

        elif op == Op.NEG:
            a = self.pop()
            if not self._is_numeric(a):
                raise RuntimeError_(f"Cannot negate {type(a).__name__}")
            self.push(-a)

        elif op == Op.NOT:
            a = self.pop()
            self.push(not self.is_truthy(a))

        elif op == Op.EQ:
            b, a = self.pop(), self.pop()
            self.push(a == b)

        elif op == Op.NE:
            b, a = self.pop(), self.pop()
            self.push(a != b)

        elif op == Op.LT:
            b, a = self.pop(), self.pop()
            self.push(a < b)

        elif op == Op.GT:
            b, a = self.pop(), self.pop()
            self.push(a > b)

        elif op == Op.LE:
            b, a = self.pop(), self.pop()
            self.push(a <= b)

        elif op == Op.GE:
            b, a = self.pop(), self.pop()
            self.push(a >= b)

        elif op == Op.AND:
            b, a = self.pop(), self.pop()
            self.push(self.is_truthy(a) and self.is_truthy(b))

        elif op == Op.OR:
            b, a = self.pop(), self.pop()
            self.push(self.is_truthy(a) or self.is_truthy(b))

        elif op == Op.LOAD:
            name = instr.arg
            if name in self.locals:
                self.push(self.locals[name])
            elif name in self.globals:
                self.push(self.globals[name])
            else:
                # Search call stack for outer scopes
                for frame in reversed(self.call_stack):
                    if name in frame.return_to_locals:
                        self.push(frame.return_to_locals[name])
                        return
                # Check if it's in globals from outer scope
                raise RuntimeError_(f"Undefined variable: {name}")

        elif op == Op.STORE:
            name = instr.arg
            value = self.pop()
            if self.call_stack:
                self.locals[name] = value
                # Also update globals if the variable exists there (for closure-like mutation)
                if name in self.globals:
                    self.globals[name] = value
            else:
                self.globals[name] = value
                self.locals[name] = value

        elif op == Op.JUMP:
            self.ip = instr.arg

        elif op == Op.JUMP_IF_FALSE:
            val = self.pop()
            if not self.is_truthy(val):
                self.ip = instr.arg

        elif op == Op.CALL:
            num_args = instr.arg
            fn = self.pop()
            if not isinstance(fn, CompiledFunction):
                raise RuntimeError_(f"Cannot call {type(fn).__name__}")
            if len(fn.params) != num_args:
                raise RuntimeError_(f"Function {fn.name} expects {len(fn.params)} arguments, got {num_args}")

            # Collect arguments (they were pushed before the function)
            args = []
            for _ in range(num_args):
                args.append(self.pop())
            args.reverse()

            # Save current frame
            frame = CallFrame(
                code=self.code,
                ip=self.ip,
                locals=self.locals,
                return_to_ip=self.ip,
                return_to_code=self.code,
                return_to_locals=self.locals,
            )
            self.call_stack.append(frame)

            # Set up new frame
            new_locals = {}
            # Copy outer variables for closure-like access
            new_locals.update(self.globals)
            for i, param in enumerate(fn.params):
                new_locals[param] = args[i]
            # Also store function itself for recursion
            new_locals[fn.name] = fn

            self.code = fn.code
            self.ip = 0
            self.locals = new_locals

        elif op == Op.RETURN:
            return_value = self.pop()
            if not self.call_stack:
                raise RuntimeError_("Return outside of function")
            frame = self.call_stack.pop()
            self.code = frame.return_to_code
            self.ip = frame.return_to_ip
            self.locals = frame.return_to_locals
            self.push(return_value)

        elif op == Op.PRINT:
            value = self.pop()
            self.output.append(self._format_value(value))

        elif op == Op.HALT:
            self.ip = len(self.code)  # Stop execution

        else:
            raise RuntimeError_(f"Unknown opcode: {op}")

    def _check_numeric(self, a, b, op_str):
        if not self._is_numeric(a) or not self._is_numeric(b):
            raise RuntimeError_(f"Cannot use '{op_str}' with {type(a).__name__} and {type(b).__name__}")

    def _format_value(self, value) -> str:
        if value is None:
            return "nil"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return str(value)
        if isinstance(value, int):
            return str(value)
        return str(value)
