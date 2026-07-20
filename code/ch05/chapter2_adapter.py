"""Import the tokenizer and decoder from Chapter 2 without duplicating them.

Chapter 2's teaching code is tangled from its ``# @save`` cells into
``code/ch02/_generated.py``. That module is loaded here under the unique name
``ch02_generated`` (several chapters each ship a ``_generated`` module, so a
bare ``sys.path`` import would collide inside one process) and the names
Chapter 5 uses are re-exported unchanged.
"""

import importlib.util
import sys
from pathlib import Path


_GENERATED = Path(__file__).resolve().parent.parent / "ch02" / "_generated.py"

if "ch02_generated" in sys.modules:
    _ch02 = sys.modules["ch02_generated"]
else:
    _spec = importlib.util.spec_from_file_location("ch02_generated", _GENERATED)
    assert _spec is not None and _spec.loader is not None
    _ch02 = importlib.util.module_from_spec(_spec)
    sys.modules["ch02_generated"] = _ch02
    _spec.loader.exec_module(_ch02)

BytePairTokenizer = _ch02.BytePairTokenizer
GPTConfig = _ch02.GPTConfig
TinyGPT = _ch02.TinyGPT


__all__ = ["BytePairTokenizer", "GPTConfig", "TinyGPT"]
