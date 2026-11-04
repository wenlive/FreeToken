"""Compatibility import for the independent test oracles."""
import sys

from tests.clover_reference import oracles as _reference

sys.modules[__name__] = _reference
