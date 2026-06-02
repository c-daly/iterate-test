import os
import sys

# Make the assignment src/ directory importable so tests can do
# `from autograd import Value` regardless of where pytest is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
