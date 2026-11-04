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


@pytest.mark.parametrize("widths,maximum,running", [([4], 32, 1), ([32], 16, 1),
    ([8], 0, 1), ([8], 32, 2)])
def test_graph_warmup_rejects_unreserved_or_unsupported_widths(widths, maximum, running):
    with pytest.raises(ValueError, match="warmup"):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16,
            max_running_req=running, clover_max_width=maximum, clover_graph_warmup_widths=widths)


def test_graph_warmup_is_opt_in_and_normalizes_declared_widths():
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16)
    assert config.clover_graph_warmup_widths == []
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16,
        max_running_req=1, clover_max_width=32, clover_graph_warmup_widths=[32, 8, 16, 8])
    assert config.clover_graph_warmup_widths == [8, 16, 32]


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


@pytest.mark.parametrize("model_type,quant,tied,expected", [
    ("clover_moe", "clover_gguf", False, True),
    ("clover_moe", "clover_gguf", True, False),
    ("clover_moe", "nvfp4", False, False),
    ("clover_moe", None, False, False),
    ("qwen3_5_moe", "nvfp4", False, False),
    ("qwen4_exp", "gguf_strata", False, False),
    ("gemma4", "gguf_q4_0", True, False),
])
def test_automatic_host_embedding_is_scoped_to_untied_clover_gguf(model_type, quant, tied, expected):
    from types import SimpleNamespace

    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16)
    config.__dict__["model_config"] = SimpleNamespace(
        model_type=model_type, expert_quant=quant, tie_word_embeddings=tied)
    assert config.host_embedding_enabled is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_embedding_placement_needs_no_automatic_model_probe(enabled):
    config = EngineConfig("nonexistent", DistributedInfo(0, 1), torch.bfloat16,
        gguf_embedding_host=enabled)
    assert config.host_embedding_enabled is enabled
    assert "hf_config" not in config.__dict__
