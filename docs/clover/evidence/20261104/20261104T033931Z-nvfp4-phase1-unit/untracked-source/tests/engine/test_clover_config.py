"""Clover reserves only the explicitly configured block capacity."""
import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig


@pytest.mark.parametrize("mode,width,maximum,expected", [
    ("causal", 8, None, 0), ("selfspec", 8, None, 8),
    ("selfspec", 16, None, 16), ("denoise", 32, None, 32),
    ("causal", 8, 32, 32), ("causal", 8, 0, 0),
])
def test_block_reserve_defaults_to_effective_engine_mode(mode, width, maximum, expected):
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16,
        max_running_req=1, clover_mode=mode, clover_width=width, clover_max_width=maximum)
    assert config.clover_max_width == expected


def test_causal_omitted_maximum_reserves_no_transactions():
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16)
    assert config.clover_max_width == 0


def test_sparse_prefill_limit_rejects_negative_values():
    with pytest.raises(ValueError, match="sparse prefill"):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, clover_sparse_prefill_limit=-1)


@pytest.mark.parametrize("limit", [0, 128, 512])
def test_runtime_configuration_applies_sparse_limit_to_all_clover_layers(limit):
    from types import SimpleNamespace
    from freetoken.models.clover_moe.model import CloverMoeModelLM

    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16,
        clover_sparse_prefill_limit=limit)
    layers = [SimpleNamespace(mlp=SimpleNamespace()) for _ in range(3)]
    model = SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=layers)))
    CloverMoeModelLM.configure_runtime(model, config)
    assert all(layer.mlp.sparse_prefill_limit == limit for layer in layers)
