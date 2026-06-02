import os
import sys

ASSIGN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ASSIGN_ROOT not in sys.path:
    sys.path.insert(0, ASSIGN_ROOT)
