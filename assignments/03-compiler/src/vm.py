"""VM: stack-based virtual machine for executing bytecode."""

from src.compiler import OpCode, CompiledProgram, CompiledFunction


class VMError(Exception):
    pass


def _format_value(value: object) -> str:
    """Format a value for print output."""
    if value is None:
        return "nil"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return str(value)
    return str(value)


def _is_truthy(value: object) -> bool:
    if value is None:
        return False
    if value is False:
        return False
    if value == 0:
        return False
    return True


class Frame:
    """Call frame for function execution."""

    def __init__(
        self,
        code,
        constants,
        locals_: dict | None = None,
        globals_: dict | None = None,
    ):
        self.code = code
        self.constants = constants
        self.ip = 0
        self.locals = locals_ if locals_ is not None else {}
        self.globals = globals_ if globals_ is not None else {}


def run(compiled_program: CompiledProgram) -> list[str]:
    output: list[str] = []
    stack: list[object] = []
    globals_: dict[str, object] = {}
    functions = compiled_program.functions

    # Create the main frame
    call_stack: list[Frame] = [
        Frame(
            compiled_program.code,
            compiled_program.constants,
            globals_=globals_,
        )
    ]

    while call_stack:
        frame = call_stack[-1]

        if frame.ip >= len(frame.code):
            break

        instr = frame.code[frame.ip]
        frame.ip += 1
        op = instr.op

        if op == OpCode.HALT:
            break

        elif op == OpCode.CONST:
            stack.append(frame.constants[instr.arg])

        elif op == OpCode.POP:
            if stack:
                stack.pop()

        elif op == OpCode.ADD:
            b = stack.pop()
            a = stack.pop()
            if isinstance(a, str) and isinstance(b, str):
                stack.append(a + b)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                stack.append(a + b)
            else:
                raise VMError(
                    f"Cannot add {type(a).__name__} and "
                    f"{type(b).__name__}"
                )

        elif op == OpCode.SUB:
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)

        elif op == OpCode.MUL:
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)

        elif op == OpCode.DIV:
            b = stack.pop()
            a = stack.pop()
            if b == 0:
                raise VMError("Division by zero")
            stack.append(a / b)

        elif op == OpCode.MOD:
            b = stack.pop()
            a = stack.pop()
            if b == 0:
                raise VMError("Modulo by zero")
            stack.append(a % b)

        elif op == OpCode.NEG:
            a = stack.pop()
            stack.append(-a)

        elif op == OpCode.NOT:
            a = stack.pop()
            stack.append(not _is_truthy(a))

        elif op == OpCode.EQ:
            b = stack.pop()
            a = stack.pop()
            stack.append(a == b)

        elif op == OpCode.NE:
            b = stack.pop()
            a = stack.pop()
            stack.append(a != b)

        elif op == OpCode.LT:
            b = stack.pop()
            a = stack.pop()
            stack.append(a < b)

        elif op == OpCode.GT:
            b = stack.pop()
            a = stack.pop()
            stack.append(a > b)

        elif op == OpCode.LE:
            b = stack.pop()
            a = stack.pop()
            stack.append(a <= b)

        elif op == OpCode.GE:
            b = stack.pop()
            a = stack.pop()
            stack.append(a >= b)

        elif op == OpCode.AND:
            b = stack.pop()
            a = stack.pop()
            stack.append(_is_truthy(a) and _is_truthy(b))

        elif op == OpCode.OR:
            b = stack.pop()
            a = stack.pop()
            stack.append(_is_truthy(a) or _is_truthy(b))

        elif op == OpCode.LOAD:
            name = instr.arg
            if name in frame.locals:
                stack.append(frame.locals[name])
            elif name in frame.globals:
                stack.append(frame.globals[name])
            elif name in functions:
                # Push function name as a callable reference
                stack.append(name)
            else:
                raise VMError(f"Undefined variable '{name}'")

        elif op == OpCode.STORE:
            name = instr.arg
            value = stack[-1] if stack else None
            if len(call_stack) == 1:
                frame.globals[name] = value
            else:
                frame.locals[name] = value

        elif op == OpCode.JUMP:
            frame.ip = instr.arg

        elif op == OpCode.JUMP_IF_FALSE:
            cond = stack.pop()
            if not _is_truthy(cond):
                frame.ip = instr.arg

        elif op == OpCode.PRINT:
            value = stack.pop()
            output.append(_format_value(value))

        elif op == OpCode.CALL:
            num_args = instr.arg
            func_ref = stack.pop()

            if not isinstance(func_ref, str):
                raise VMError(f"Cannot call {type(func_ref).__name__}")

            if func_ref not in functions:
                raise VMError(f"Undefined function '{func_ref}'")

            fn: CompiledFunction = functions[func_ref]
            if len(fn.params) != num_args:
                raise VMError(
                    f"Function '{func_ref}' expects {len(fn.params)} "
                    f"args, got {num_args}"
                )

            # Pop args from stack (they were pushed in order)
            args = []
            for _ in range(num_args):
                args.append(stack.pop())
            args.reverse()

            fn_locals = {}
            for param, val in zip(fn.params, args):
                fn_locals[param] = val

            new_frame = Frame(
                fn.code,
                fn.constants,
                locals_=fn_locals,
                globals_=globals_,
            )
            call_stack.append(new_frame)

        elif op == OpCode.RETURN:
            ret_val = stack.pop() if stack else None
            call_stack.pop()
            stack.append(ret_val)

    return output
