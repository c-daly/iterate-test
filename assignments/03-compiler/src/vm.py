"""Stack-based virtual machine that executes compiled bytecode.

Execution model
---------------
The VM keeps an operand stack and a stack of call frames. Each frame holds:
  - the chunk (instruction list) being executed,
  - an instruction pointer (ip),
  - a *scope* (dict of local variable bindings),
  - the operand-stack height to truncate back to on return.

Variable resolution uses a scope chain: every function frame gets a fresh
scope whose enclosing scope is the *global* scope, so a function has its own
locals/parameters but can still read outer (top-level) variables. Top-level
code runs in the global scope directly.

The VM raises VMError for runtime faults required by the spec: undefined
variables, division/modulo by zero, type errors, and arity mismatches.
"""

from src.compiler import Function


class VMError(Exception):
    """Runtime fault raised during bytecode execution."""


class Scope:
    """A single lexical scope; chains to an enclosing scope for reads."""

    __slots__ = ("vars", "parent")

    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def define(self, name, value):
        self.vars[name] = value

    def get(self, name):
        scope = self
        while scope is not None:
            if name in scope.vars:
                return scope.vars[name]
            scope = scope.parent
        raise VMError("undefined variable: %s" % name)

    def set(self, name, value):
        scope = self
        while scope is not None:
            if name in scope.vars:
                scope.vars[name] = value
                return
            scope = scope.parent
        raise VMError("undefined variable: %s" % name)


class Frame:
    __slots__ = ("chunk", "ip", "scope", "base")

    def __init__(self, chunk, scope, base):
        self.chunk = chunk
        self.ip = 0
        self.scope = scope
        self.base = base


NUMERIC = (int, float)


def _is_number(v):
    # bool is a subclass of int but should not be treated as numeric here.
    return isinstance(v, NUMERIC) and not isinstance(v, bool)


class VM:
    def __init__(self):
        self.stack = []
        self.frames = []
        self.output = []
        self.global_scope = Scope()

    # --- helpers -----------------------------------------------------------
    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if not self.stack:
            raise VMError("stack underflow")
        return self.stack.pop()

    def run(self, chunk):
        """Execute a top-level chunk and return the list of printed values."""
        frame = Frame(chunk, self.global_scope, 0)
        self.frames.append(frame)

        while self.frames:
            frame = self.frames[-1]
            if frame.ip >= len(frame.chunk):
                # Fell off the end of a chunk without an explicit terminator.
                self.frames.pop()
                continue
            instr = frame.chunk[frame.ip]
            frame.ip += 1
            op = instr[0]
            handler = self._DISPATCH.get(op)
            if handler is None:
                raise VMError("unknown opcode: %s" % op)
            result = handler(self, frame, instr)
            if result is _HALT:
                break
        return self.output

    # --- opcode handlers ---------------------------------------------------
    def op_const(self, frame, instr):
        self.push(instr[1])

    def op_pop(self, frame, instr):
        self.pop()

    def op_dup(self, frame, instr):
        self.push(self.stack[-1])

    def op_add(self, frame, instr):
        b = self.pop()
        a = self.pop()
        if _is_number(a) and _is_number(b):
            self.push(a + b)
        elif isinstance(a, str) and isinstance(b, str):
            self.push(a + b)
        else:
            raise VMError("type error: cannot add %s and %s"
                          % (_typename(a), _typename(b)))

    def op_sub(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self._require_numbers(a, b, "subtract")
        self.push(a - b)

    def op_mul(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self._require_numbers(a, b, "multiply")
        self.push(a * b)

    def op_div(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self._require_numbers(a, b, "divide")
        if b == 0:
            raise VMError("division by zero")
        if isinstance(a, int) and isinstance(b, int):
            # int / int stays int, truncating toward zero without float
            # conversion (preserves arbitrary precision for ints > 2^53).
            self.push(a // b if (a < 0) == (b < 0) else -(-a // b))
        else:
            self.push(a / b)

    def op_mod(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self._require_numbers(a, b, "modulo")
        if b == 0:
            raise VMError("modulo by zero")
        if isinstance(a, int) and isinstance(b, int):
            # C-style remainder consistent with truncating op_div, so the
            # invariant a == (a // b) * b + (a % b) holds for negatives too.
            q = a // b if (a < 0) == (b < 0) else -(-a // b)
            self.push(a - q * b)
        else:
            self.push(a % b)

    def op_neg(self, frame, instr):
        a = self.pop()
        if not _is_number(a):
            raise VMError("type error: cannot negate %s" % _typename(a))
        self.push(-a)

    def op_not(self, frame, instr):
        a = self.pop()
        self.push(not _truthy(a))

    def op_eq(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self.push(_equal(a, b))

    def op_ne(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self.push(not _equal(a, b))

    def op_lt(self, frame, instr):
        self._compare("<")

    def op_gt(self, frame, instr):
        self._compare(">")

    def op_le(self, frame, instr):
        self._compare("<=")

    def op_ge(self, frame, instr):
        self._compare(">=")

    def op_and(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self.push(_truthy(a) and _truthy(b))

    def op_or(self, frame, instr):
        b = self.pop()
        a = self.pop()
        self.push(_truthy(a) or _truthy(b))

    def op_load(self, frame, instr):
        self.push(frame.scope.get(instr[1]))

    def op_define(self, frame, instr):
        value = self.pop()
        frame.scope.define(instr[1], value)

    def op_store(self, frame, instr):
        # Assignment is an expression: leave the value on the stack.
        value = self.stack[-1]
        frame.scope.set(instr[1], value)

    def op_jump(self, frame, instr):
        frame.ip = instr[1]

    def op_jump_if_false(self, frame, instr):
        cond = self.pop()
        if not _truthy(cond):
            frame.ip = instr[1]

    def op_jump_if_true(self, frame, instr):
        cond = self.pop()
        if _truthy(cond):
            frame.ip = instr[1]

    def op_call(self, frame, instr):
        argc = instr[1]
        args = [self.pop() for _ in range(argc)]
        args.reverse()
        callee = self.pop()
        if not isinstance(callee, Function):
            raise VMError("type error: %s is not callable" % _typename(callee))
        if callee.arity != argc:
            raise VMError("arity error: %s expects %d args, got %d"
                          % (callee.name, callee.arity, argc))
        # Fresh scope chained to globals: own locals, can read outer vars.
        scope = Scope(self.global_scope)
        for name, value in zip(callee.params, args):
            scope.define(name, value)
        new_frame = Frame(callee.chunk, scope, len(self.stack))
        self.frames.append(new_frame)

    def op_return(self, frame, instr):
        value = self.pop()
        finished = self.frames.pop()
        # Drop anything the callee left on the operand stack.
        del self.stack[finished.base:]
        if self.frames:
            self.push(value)
        # If no frames remain a bare top-level return just ends; ignore value.

    def op_print(self, frame, instr):
        self.output.append(self.pop())

    def op_halt(self, frame, instr):
        return _HALT

    # --- shared logic ------------------------------------------------------
    def _require_numbers(self, a, b, what):
        if not (_is_number(a) and _is_number(b)):
            raise VMError("type error: cannot %s %s and %s"
                          % (what, _typename(a), _typename(b)))

    def _compare(self, op):
        b = self.pop()
        a = self.pop()
        if _is_number(a) and _is_number(b):
            pass
        elif isinstance(a, str) and isinstance(b, str):
            pass
        else:
            raise VMError("type error: cannot compare %s and %s"
                          % (_typename(a), _typename(b)))
        if op == "<":
            self.push(a < b)
        elif op == ">":
            self.push(a > b)
        elif op == "<=":
            self.push(a <= b)
        else:
            self.push(a >= b)


_HALT = object()


def _truthy(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if _is_number(value):
        return value != 0
    if isinstance(value, str):
        return len(value) > 0
    return True


def _equal(a, b):
    # Keep bool and number distinct from each other for == semantics.
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or (isinstance(a, bool) and isinstance(b, bool) and a == b)
    if _is_number(a) and _is_number(b):
        return a == b
    if type(a) is type(b):
        return a == b
    return False


def _typename(v):
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, Function):
        return "function"
    return type(v).__name__


# Dispatch table built once from the op_* methods.
VM._DISPATCH = {
    "CONST": VM.op_const,
    "POP": VM.op_pop,
    "DUP": VM.op_dup,
    "ADD": VM.op_add,
    "SUB": VM.op_sub,
    "MUL": VM.op_mul,
    "DIV": VM.op_div,
    "MOD": VM.op_mod,
    "NEG": VM.op_neg,
    "NOT": VM.op_not,
    "EQ": VM.op_eq,
    "NE": VM.op_ne,
    "LT": VM.op_lt,
    "GT": VM.op_gt,
    "LE": VM.op_le,
    "GE": VM.op_ge,
    "AND": VM.op_and,
    "OR": VM.op_or,
    "LOAD": VM.op_load,
    "DEFINE": VM.op_define,
    "STORE": VM.op_store,
    "JUMP": VM.op_jump,
    "JUMP_IF_FALSE": VM.op_jump_if_false,
    "JUMP_IF_TRUE": VM.op_jump_if_true,
    "CALL": VM.op_call,
    "RETURN": VM.op_return,
    "PRINT": VM.op_print,
    "HALT": VM.op_halt,
}
