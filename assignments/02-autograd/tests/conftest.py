"""Pytest config: expose the src/ package to tests under top-level names.

Makes both `from autograd import Value` and `from nn import MLP` resolve
while still allowing nn.py to use the relative import `from .autograd`
specified by the assignment.
"""
import sys
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))

# Importing as a package preserves the relative-import semantics in nn.py.
import src.autograd as _autograd  # noqa: E402
import src.nn as _nn  # noqa: E402

sys.modules.setdefault("autograd", _autograd)
sys.modules.setdefault("nn", _nn)
