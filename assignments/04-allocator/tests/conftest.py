import os
import sys

# Make the assignment src directory importable as top-level modules
# (allocator, buddy, firstfit, compare).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
