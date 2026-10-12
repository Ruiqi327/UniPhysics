"""Compatibility entry point; implementation lives in uniphysics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uniphysics.cli import main

if __name__ == "__main__":
    main(['local'] + sys.argv[1:])
