"""Compatibility import and command for the bounded official HF reference."""
import sys

from tests.clover_reference import hf as _reference

if __name__ == "__main__":
    _reference.main()
else:
    sys.modules[__name__] = _reference
