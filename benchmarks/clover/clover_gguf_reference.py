"""Compatibility alias; A/B tools patch the canonical reference module's globals."""
import sys

from tests.clover_reference import gguf as _reference

sys.modules[__name__] = _reference
