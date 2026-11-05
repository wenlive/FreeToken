"""Test native wheel selection without importing CUDA or initializing a device."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


def isolated_loader(monkeypatch, native=None):
    # Avoid the kernel package's eager GPU imports: only exercise this loader.
    package = ModuleType('gguf_loader_case')
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, 'torch', ModuleType('torch'))
    if native is not None:
        monkeypatch.setitem(sys.modules, 'gguf_loader_case._gguf', native)
    path = Path(__file__).parents[2]/'python/freetoken/kernel/gguf.py'
    spec = importlib.util.spec_from_file_location('gguf_loader_case.gguf', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wheel_module_loads_even_when_native_jit_is_disabled(monkeypatch):
    native = ModuleType('gguf_loader_case._gguf')
    monkeypatch.setenv('FREETOKEN_DISABLE_JIT', '1')
    assert isolated_loader(monkeypatch, native)._module() is native


@pytest.mark.parametrize('allow', ['0', '1'])
def test_disable_jit_cannot_be_overridden_by_development_opt_in(monkeypatch, allow):
    monkeypatch.setenv('FREETOKEN_DISABLE_JIT', '1')
    monkeypatch.setenv('FREETOKEN_GGUF_ALLOW_JIT', allow)
    with pytest.raises(RuntimeError, match='native JIT is disabled'):
        isolated_loader(monkeypatch)._module()


def test_missing_wheel_requires_explicit_development_jit(monkeypatch):
    monkeypatch.delenv('FREETOKEN_DISABLE_JIT', raising=False)
    monkeypatch.delenv('FREETOKEN_GGUF_ALLOW_JIT', raising=False)
    with pytest.raises(RuntimeError, match='matching FreeToken wheel'):
        isolated_loader(monkeypatch)._module()
