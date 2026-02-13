class VMError(Exception):
    pass


class Function:
    def __init__(self, name, params, code):
        self.name = name
        self.params = params
        self.code = code

    def __repr__(self):
        return f"<Function {self.name}>"


class CallFrame:
    def __init__(self, code, locals_, return_addr):
        self.code = code
        self.locals = locals_
        self.ip = 0
        self.return_addr = return_addr


def _is_falsy(val):
    return val is False or val is None or val == 0 or val == 0.0 or val == ""


def _format_value(val):
    if val is True:
        return "true"
    if val is False:
        return "false"
    if val is None:
        return "nil"
    return str(val)


class VM:
    def __init__(self, code):
        self.stack = []
        self.output = []
        self.globals = {}
        self.frames = [CallFrame(code, {}, -1)]

    def run(self) -> list:
        from compiler import OpCode

        while True:
            frame = self.frames[-1]
            if frame.ip >= len(frame.code):
                break
            instr = frame.code[frame.ip]
            frame.ip += 1
            op = instr.op

            if op == OpCode.CONST:
                self.stack.append(instr.arg)

            elif op == OpCode.POP:
                self.stack.pop()

            elif op == OpCode.ADD:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, str) and isinstance(b, str):
                    self.stack.append(a + b)
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    result = a + b
                    if isinstance(a, int) and isinstance(b, int):
                        self.stack.append(result)
                    else:
                        self.stack.append(float(result))
                else:
                    raise VMError(f"Cannot add {type(a).__name__} and {type(b).__name__}")

            elif op == OpCode.SUB:
                b = self.stack.pop()
                a = self.stack.pop()
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise VMError(f"Cannot subtract {type(a).__name__} and {type(b).__name__}")
                result = a - b
                if isinstance(a, int) and isinstance(b, int):
                    self.stack.append(result)
                else:
                    self.stack.append(float(result))

            elif op == OpCode.MUL:
                b = self.stack.pop()
                a = self.stack.pop()
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise VMError(f"Cannot multiply {type(a).__name__} and {type(b).__name__}")
                result = a * b
                if isinstance(a, int) and isinstance(b, int):
                    self.stack.append(result)
                else:
                    self.stack.append(float(result))

            elif op == OpCode.DIV:
                b = self.stack.pop()
                a = self.stack.pop()
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise VMError(f"Cannot divide {type(a).__name__} and {type(b).__name__}")
                if b == 0:
                    raise VMError("Division by zero")
                result = a / b
                if isinstance(a, int) and isinstance(b, int) and a % b == 0:
                    self.stack.append(int(result))
                else:
                    self.stack.append(float(result))

            elif op == OpCode.MOD:
                b = self.stack.pop()
                a = self.stack.pop()
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise VMError(f"Cannot modulo {type(a).__name__} and {type(b).__name__}")
                if b == 0:
                    raise VMError("Division by zero")
                result = a % b
                if isinstance(a, int) and isinstance(b, int):
                    self.stack.append(result)
                else:
                    self.stack.append(float(result))

            elif op == OpCode.NEG:
                val = self.stack.pop()
                if not isinstance(val, (int, float)):
                    raise VMError(f"Cannot negate {type(val).__name__}")
                self.stack.append(-val)

            elif op == OpCode.NOT:
                val = self.stack.pop()
                self.stack.append(_is_falsy(val))

            elif op == OpCode.EQ:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a == b)

            elif op == OpCode.NE:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a != b)

            elif op == OpCode.LT:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a < b)

            elif op == OpCode.GT:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a > b)

            elif op == OpCode.LE:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a <= b)

            elif op == OpCode.GE:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a >= b)

            elif op == OpCode.AND:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a if _is_falsy(a) else b)

            elif op == OpCode.OR:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a if not _is_falsy(a) else b)

            elif op == OpCode.LOAD:
                name = instr.arg
                if name in frame.locals:
                    self.stack.append(frame.locals[name])
                elif name in self.globals:
                    self.stack.append(self.globals[name])
                else:
                    raise VMError(f"Undefined variable: {name}")

            elif op == OpCode.STORE:
                name = instr.arg
                val = self.stack.pop()
                if len(self.frames) > 1:
                    frame.locals[name] = val
                else:
                    self.globals[name] = val

            elif op == OpCode.JUMP:
                frame.ip = instr.arg

            elif op == OpCode.JUMP_IF_FALSE:
                val = self.stack.pop()
                if _is_falsy(val):
                    frame.ip = instr.arg

            elif op == OpCode.PRINT:
                val = self.stack.pop()
                self.output.append(_format_value(val))

            elif op == OpCode.CALL:
                nargs = instr.arg
                func = self.stack.pop()
                if not isinstance(func, Function):
                    raise VMError(f"Cannot call {type(func).__name__}")
                if len(func.params) != nargs:
                    raise VMError(f"{func.name} expects {len(func.params)} args, got {nargs}")
                args = []
                for _ in range(nargs):
                    args.append(self.stack.pop())
                args.reverse()
                locals_ = dict(zip(func.params, args))
                self.frames.append(CallFrame(func.code, locals_, -1))

            elif op == OpCode.RETURN:
                return_val = self.stack.pop()
                self.frames.pop()
                self.stack.append(return_val)

            elif op == OpCode.HALT:
                return self.output

            else:
                raise VMError(f"Unknown opcode: {op}")

        return self.output
