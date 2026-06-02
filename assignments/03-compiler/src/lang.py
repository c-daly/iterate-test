"""Convenience entry point tying the whole pipeline together.

    execute(source) = lex -> parse -> compile -> run, returning the list of
    values handed to `print`.

Values are returned as-is (int / float / str / bool / None), not stringified,
so callers and tests can assert on the actual Python values.
"""

from src.compiler import compile_source
from src.vm import VM


def execute(source):
    """Compile and run `source`, returning the list of printed values."""
    chunk = compile_source(source)
    vm = VM()
    return vm.run(chunk)
