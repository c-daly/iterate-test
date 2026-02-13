"""Add src/ to sys.path so tests can import skiplist directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
