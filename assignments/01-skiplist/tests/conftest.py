import os
import sys

# Make src/ importable regardless of the directory pytest is invoked from.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)
