"""Silent layout, immutable-config and token-boundary contracts for Clover GGUF."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.clover_moe import gguf as adapter
from freetoken.models.gguf.config import GgufConfigShim


def test_aot_claims_gguf_architecture_and_both_actual_expert_rows():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS, expert_bank_row_bytes

    entry = next(
        model
        for model in SUPPORTED_MODELS
        if "CloverMoeGGUFModelLM" in model.arch_aliases
    )
    assert adapter.EXPERT_FORMAT in entry.expert_formats
    rows = expert_bank_row_bytes(
        adapter.EXPERT_FORMAT, entry.hidden_size, entry.moe_intermediate_size
    )
    assert rows == {"gate_up": 2_228_224, "down": 1_114_112}
    assert sum(rows.values()) == 3_342_336
    mixed = expert_bank_row_bytes(adapter.EXPERT_FORMAT, entry.hidden_size,
        entry.moe_intermediate_size, gguf_types=(23, 12))
    assert mixed == {"gate_up": 1_114_112, "down": 589_824}
    from freetoken.kernel.aot_models import fast_index_copy_feature_sizes

    assert set(rows.values()) | set(mixed.values()) <= fast_index_copy_feature_sizes(entry)


@pytest.fixture
def shim(monkeypatch):
    metadata = {
        "block_count": 4,
        "embedding_length": 256,
        "context_length": 1024,
        "attention.head_count": 4,
        "attention.head_count_kv": 2,
        "attention.key_length": 64,
        "attention.layer_norm_rms_epsilon": 1e-6,
        "attention.recurrent_layers": [True, True, True, False],
        "expert_count": 4,
        "expert_used_count": 2,
        "expert_feed_forward_length": 256,
        "expert_shared_feed_forward_length": 256,
        "ssm.group_count": 2,
        "ssm.time_step_rank": 4,
        "ssm.inner_size": 256,
        "ssm.state_size": 64,
        "ssm.conv_kernel": 4,
        "rope.dimension_count": 16,
        "rope.freq_base": 10000.0,
    }
    tensors = [
        SimpleNamespace(
            name=f"blk.{layer}.ffn_{role}_exps.weight",
            shape=(4, 256, 256),
            ggml_type=12 if role == "down" else 23,
            row_bytes=144 if role == "down" else 136,
        )
        for layer in range(4)
        for role in ("gate", "up", "down")
    ]
    monkeypatch.setattr(adapter, "iter_gguf_tensors", lambda _: iter(tensors))
    value = GgufConfigShim(
        ["CloverMoeGGUFModelLM"],
        "synthetic.gguf",
        "clover_moe",
        {"clover_moe." + key: item for key, item in metadata.items()},
        512,
        False,
    )
    return value, tensors


def test_gguf_geometry_and_exact_expert_footprint(shim):
    from freetoken.moe.expert_banks import bank_bytes_estimate

    value, _ = shim
    config = adapter.parse_gguf_config(value)
    assert config.expert_quant == "clover_gguf"
    assert config.linear_attention_group().layer_ids == (0, 1, 2)
    assert config.attention_group_for_layer(3).layer_ids == (3,)
    assert config.rotary_config.rotary_dim == 16
    assert bank_bytes_estimate(config) == 4 * 4 * (512 * 136 + 256 * 144)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "type", "shape", "layer"])
def test_incomplete_or_incompatible_experts_fail_before_allocation(shim, damage):
    value, tensors = shim
    if damage == "missing":
        tensors.pop()
    elif damage == "duplicate":
        tensors.append(tensors[0])
    elif damage == "type":
        tensors[0].ggml_type = 99
    elif damage == "shape":
        tensors[0].shape = (4, 128, 256)
    else:
        tensors[0].name = "blk.9.ffn_gate_exps.weight"
    with pytest.raises(ValueError):
        adapter.parse_gguf_config(value)


@pytest.mark.parametrize("axis", [0, 1])
def test_cyclic_head_order_is_inverted_with_asymmetric_groups(axis):
    nk, repeat, dim = 16, 2, 3
    original = torch.arange(nk * repeat * dim * 5).reshape(nk * repeat * dim, 5)
    # Independent labels define the exporter's order; applying the forward
    # permutation a second time would fail for these asymmetric dimensions.
    labels = [head * repeat + lane for lane in range(repeat) for head in range(nk)]
    indices = torch.tensor([label * dim + j for label in labels for j in range(dim)])
    exported = original[indices]
    if axis:
        original, exported = original.T, exported.T
    restored = adapter.restore_v_heads(exported, axis, nk, nk * repeat, dim)
    torch.testing.assert_close(restored, original, rtol=0, atol=0)


def test_expert_types_are_read_per_layer_and_role(shim):
    import gguf

    value, tensors = shim
    expected = [(8, 8), (12, 14), (2, 2), (14, 8)]
    for tensor in tensors:
        layer = int(tensor.name.split(".")[1])
        kind = expected[layer][int("down" in tensor.name)]
        block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
        tensor.ggml_type = kind
        tensor.row_bytes = 256 // block * size
    config = adapter.parse_gguf_config(value)
    assert config.expert_source["types"] == expected
    assert config.expert_source["row_bytes"][1] == (144, 210)
    from freetoken.moe.expert_banks import bank_bytes_estimate
    # A unified slot pool reserves the largest packed row of each role.
    assert bank_bytes_estimate(config) == 4 * 4 * (512 + 256) * 272


def test_streaming_mixed_expert_loader_preserves_bytes_and_releases_mapping(tmp_path, monkeypatch):
    import gguf
    import numpy as np

    path = tmp_path / "mixed.gguf"
    writer = gguf.GGUFWriter(str(path), "clover_moe")
    originals = {}
    for layer, kinds in enumerate(((8, 8), (12, 14))):
        for role in ("gate", "up", "down"):
            kind = kinds[int(role == "down")]
            block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
            raw = (np.arange(2 * 256 * (256 // block * size), dtype=np.uint32) % 251).astype(np.uint8).reshape(2, 256, -1)
            originals[layer, role] = raw.copy()
            writer.add_tensor(f"blk.{layer}.ffn_{role}_exps.weight", raw,
                              raw_dtype=gguf.GGMLQuantizationType(kind))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = SimpleNamespace(num_layers=2, num_experts=2, hidden_size=256, moe_intermediate_size=256)
    banks = adapter.load_packed_expert_sources(str(path), config)
    # The file can disappear after load: serving owns only independent host bytes.
    path.unlink()
    for (layer, role), raw in originals.items():
        bank = banks["down" if role == "down" else "gate_up"][layer]
        rb = raw.shape[-1]
        offset = 256 * rb if role == "up" else 0
        actual = bank.as_strided(raw.shape, (bank.stride(0), rb, 1), offset)
        np.testing.assert_array_equal(actual.numpy(), raw)


def test_raw_alog_is_restored_without_exponentiation():
    group = SimpleNamespace(
        num_key_heads=2, num_value_heads=4, key_head_dim=2, value_head_dim=2
    )
    config = SimpleNamespace(linear_attention_group=lambda: group)
    exported = torch.tensor([-3.0, -1.0, 2.0, 4.0])
    actual = adapter._restore_dense(
        "model.layers.0.linear_attn.A_log", exported, config
    )
    torch.testing.assert_close(actual, torch.tensor([-3.0, 2.0, -1.0, 4.0]).bfloat16())


def test_norm_unfolding_preserves_effective_fp32_weights():
    group = SimpleNamespace(
        num_key_heads=2, num_value_heads=4, key_head_dim=2, value_head_dim=2
    )
    config = SimpleNamespace(linear_attention_group=lambda: group)
    delta = torch.tensor([0.125, -0.25, 1e-8, 2.0]).bfloat16()
    exported = 1.0 + delta.float()
    restored = adapter._restore_dense(
        "model.layers.0.input_layernorm.weight", exported, config
    )
    torch.testing.assert_close(restored.float() + 1.0, exported, rtol=0, atol=0)


def test_engine_accepts_frozen_gguf_config_without_mutating_encoder_fields(shim):
    from freetoken.engine.config import EngineConfig

    value, _ = shim
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    engine = EngineConfig(
        dtype=torch.bfloat16,
        model_path="synthetic.gguf",
        tp_info=try_get_tp_info(),
        moe_strategy="offload",
    )
    engine.__dict__["hf_config"] = value
    assert engine.model_config.model_type == "clover_moe"
    assert not hasattr(value, "vision_config")


@pytest.mark.parametrize("strategy", ["cpu", "hybrid", "fused"])
def test_unsupported_expert_execution_is_rejected_early(shim, strategy):
    from freetoken.engine.config import EngineConfig

    value, _ = shim
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    engine = EngineConfig(
        dtype=torch.bfloat16,
        model_path="synthetic.gguf",
        tp_info=try_get_tp_info(),
        moe_strategy=strategy,
    )
    engine.__dict__["hf_config"] = value
    with pytest.raises(ValueError, match="GPU expert offload"):
        _ = engine.model_config


def test_invalid_recurrent_pattern_is_rejected(shim):
    value, _ = shim
    broken = replace(
        value,
        metadata={**value.metadata, "clover_moe.attention.recurrent_layers": [1] * 4},
    )
    with pytest.raises(ValueError, match="one bool per layer"):
        adapter.parse_gguf_config(broken)


def test_clover_control_and_user_tokens_keep_ids_and_stop_boundaries(monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from freetoken.models.gguf import tokenizer as module

    tokens = [
        "a",
        "<|endoftext|>",
        "</think>",
        "<tool_call>",
        "<|im_end|>",
        "<|mask|>",
        "<|im_start|>",
    ]
    metadata = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [1, 3, 4, 4, 3, 3, 3],
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 1,
        "tokenizer.ggml.eos_token_id": 4,
        "tokenizer.ggml.mask_token_id": 5,
        "tokenizer.ggml.normalizer.nfc": True,
    }
    monkeypatch.setattr(module, "load_gguf_metadata", lambda _: metadata)
    monkeypatch.setattr(module, "gguf_architecture", lambda _: "clover_moe")
    import transformers.integrations.ggml as converter

    def convert(arch, data):
        assert arch == "qwen2"
        return Tokenizer(BPE({token: i for i, token in enumerate(tokens)}, [])), {}

    monkeypatch.setattr(converter, "convert_gguf_tokenizer", convert)
    tokenizer = module.load_gguf_tokenizer("synthetic.gguf")
    assert len(tokenizer) == len(tokens)
    assert tokenizer.encode(
        "</think><tool_call><|mask|>", add_special_tokens=False
    ) == [2, 3, 5]
    assert (
        tokenizer.decode([2, 3, 5], skip_special_tokens=True) == "</think> <tool_call>"
    )
    assert module.gguf_eos_token_ids("synthetic.gguf", tokenizer) == {1, 4}


def test_gguf_sampling_metadata_overrides_published_clover_defaults():
    from freetoken.models.clover_moe.generation import load_generation_settings

    config = SimpleNamespace(
        metadata={
            "tokenizer.ggml.tokens": ["</think>"],
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 3,
            "tokenizer.ggml.mask_token_id": 4,
            "general.sampling.penalty_present": 1.2,
            "general.sampling.penalty_repeat": 1.1,
        }
    )
    mask, generation = load_generation_settings("synthetic.gguf", config)
    assert mask == 4 and generation["presence_penalty"] == 1.2
    assert generation["repetition_penalty"] == 1.1
    assert generation["dllm_length_penalty"] == 10.0
    assert generation["dllm_think_end_token_id"] == 0
    assert generation["eos_token_id"] == [2, 3]
    from freetoken.models.clover_moe.generation import predict_with_confidence

    logits = torch.tensor([[0.0, 2.0, 0.0, -1.0, 0.0]])
    prediction, confidence = predict_with_confidence(
        logits, SimpleNamespace(**generation)
    )
    assert prediction.tolist() == [1]
    torch.testing.assert_close(confidence, logits.softmax(-1).max(-1).values)
