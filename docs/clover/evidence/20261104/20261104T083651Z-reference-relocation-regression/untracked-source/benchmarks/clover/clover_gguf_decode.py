"""Compatibility import for the independent GGUF decoder."""
import sys

from tests.clover_reference import gguf_decode as _reference

sys.modules[__name__] = _reference
