"""Clover GGUF adapter with native mixed IQ4_XS/Q4_K expert banks.

Layout reference: ggml-org/llama.cpp@4838ffb0c8ade92f34d9157deea38d92758ab824,
conversion/clover.py and conversion/qwen.py (MIT). This adapter reverses the
exported cyclic V-head order; causal/preview operators remain FreeToken's.
"""

from dataclasses import replace
import re
from types import SimpleNamespace

import torch

from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.models.qwen3_5_moe.config import parse_config as parse_qwen
from freetoken.models.qwen3_5_moe.weight import _DenseReader, _is_gemma_norm
from freetoken.models.register import get_model_spec

from .model import CloverMoeModelLM as CloverMoeModelLM

EXPERT_FORMAT = "clover_gguf"
GATE_UP_TYPE, DOWN_TYPE = 23, 12  # IQ4_XS, Q4_K
_EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
_SCALARS = {
    "ssm_a": "A_log",
    "ssm_dt.bias": "dt_bias",
    "ssm_bd_history.bias": "bd_softmax_history_bias",
    "ssm_bd_history.weight": "bd_softmax_history_c",
    "ssm_bd_logit_scale.weight": "bd_softmax_logit_scale",
}
_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_conv1d.weight_right": "linear_attn.conv_right_weight",
    **{key: "linear_attn." + value for key, value in _SCALARS.items()},
}


def _require_metadata(shim, key):
    name = "clover_moe." + key
    if name not in shim.metadata:
        raise ValueError(f"Clover GGUF is missing {name}")
    return shim.metadata[name]


def _validate_experts(path, config):
    expected = {
        (layer, role)
        for layer in range(config.num_layers)
        for role in ("gate", "up", "down")
    }
    seen = set()
    for tensor in iter_gguf_tensors(path):
        match = _EXPERT.fullmatch(tensor.name)
        if not match:
            continue
        layer, role = int(match[1]), match[2]
        key = layer, role
        if key not in expected or key in seen:
            raise ValueError(
                f"Unexpected or duplicate Clover expert tensor {tensor.name}"
            )
        h, i, e = config.hidden_size, config.moe_intermediate_size, config.num_experts
        shape = (e, h, i) if role == "down" else (e, i, h)
        kind = DOWN_TYPE if role == "down" else GATE_UP_TYPE
        if tensor.shape != shape or tensor.ggml_type != kind:
            raise ValueError(
                f"{tensor.name}: expected shape {shape}, GGML type {kind}; "
                f"got {tensor.shape}, type {tensor.ggml_type}. This adapter serves "
                "IQ4_XS gate/up and Q4_K down without requantizing."
            )
        seen.add(key)
    if seen != expected:
        raise ValueError(
            f"Missing Clover GGUF expert tensors: {sorted(expected - seen)[:8]}"
        )


def parse_gguf_config(shim):
    def g(key):
        return _require_metadata(shim, key)

    layers = int(g("block_count"))
    recurrent = g("attention.recurrent_layers")
    if len(recurrent) != layers or any(type(item) is not bool for item in recurrent):
        raise ValueError("Clover GGUF recurrent_layers must contain one bool per layer")
    nk, nv = int(g("ssm.group_count")), int(g("ssm.time_step_rank"))
    inner, dk = int(g("ssm.inner_size")), int(g("ssm.state_size"))
    head = int(g("attention.key_length"))
    if nk <= 0 or nv <= 0 or nv % nk or inner % nv:
        raise ValueError("Invalid Clover GDN head geometry")
    hf = SimpleNamespace(
        model_type="clover_moe",
        architectures=shim.architectures,
        hidden_size=int(g("embedding_length")),
        vocab_size=shim.vocab_size,
        num_hidden_layers=layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=head,
        max_position_embeddings=int(g("context_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        tie_word_embeddings=shim.tie_word_embeddings,
        linear_num_key_heads=nk,
        linear_num_value_heads=nv,
        linear_key_head_dim=dk,
        linear_value_head_dim=inner // nv,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")),
        layer_types=["linear_attention" if r else "full_attention" for r in recurrent],
        rope_parameters=dict(
            rope_type="default",
            rope_theta=float(g("rope.freq_base")),
            partial_rotary_factor=int(g("rope.dimension_count")) / head,
        ),
    )
    config = replace(
        parse_qwen(hf), expert_quant=EXPERT_FORMAT, moe_weight_format=EXPERT_FORMAT
    )
    _validate_experts(shim.model_path, config)
    return config


def restore_v_heads(value, axis, nk, nv, head_dim):
    """Inverse of the exporter's [K, V/K, D] -> [V/K, K, D] permutation."""
    axis %= value.ndim
    if value.shape[axis] != nv * head_dim:
        raise ValueError("Clover V-head axis does not match the declared geometry")
    shape = list(value.shape)
    split = shape[:axis] + [nv // nk, nk, head_dim] + shape[axis + 1 :]
    return value.reshape(split).transpose(axis, axis + 1).reshape(shape).contiguous()


def _restore_dense(name, value, config):
    group = config.linear_attention_group()
    nk, nv, dk, dv = (
        group.num_key_heads,
        group.num_value_heads,
        group.key_head_dim,
        group.value_head_dim,
    )
    if ".linear_attn." in name:
        suffix = name.split(".linear_attn.", 1)[1]
        if suffix in ("in_proj_qkv.weight", "conv1d.weight", "conv_right_weight"):
            boundary = 2 * nk * dk
            value = torch.cat(
                (value[:boundary], restore_v_heads(value[boundary:], 0, nk, nv, dv))
            )
            if suffix in ("conv1d.weight", "conv_right_weight"):
                value = value.unsqueeze(1)
        elif suffix == "in_proj_z.weight":
            value = restore_v_heads(value, 0, nk, nv, dv)
        elif suffix in ("in_proj_a.weight", "in_proj_b.weight", *_SCALARS.values()):
            value = restore_v_heads(value, 0, nk, nv, 1)
        elif suffix == "out_proj.weight":
            value = restore_v_heads(value, 1, nk, nv, dv)
        # Clover exports raw A_log, not Qwen's -exp(A_log).
    if _is_gemma_norm(name):
        value = value.float() - 1.0
    if name.endswith(".shared_expert_gate.weight"):
        value = value.reshape(1, config.hidden_size)
    return value.to(torch.bfloat16)


def iter_hf_gguf_weights(model_path, config):
    """Dense tensors in original Clover key/head order for independent HF oracles."""
    import gguf
    import numpy as np

    global_names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "output_norm.weight": "model.norm.weight",
    }
    for tensor in iter_gguf_tensors(model_path):
        if _EXPERT.fullmatch(tensor.name):
            continue
        name = global_names.get(tensor.name)
        if name is None:
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
            if match is None or match[2] not in _LAYER:
                raise ValueError(f"Unmapped Clover GGUF tensor {tensor.name}")
            name = f"model.layers.{int(match[1])}.{_LAYER[match[2]]}"
        # The CPU reader is independent of the CUDA inference dequant kernels.
        raw = tensor._raw
        values = gguf.dequantize(raw, gguf.GGMLQuantizationType(tensor.ggml_type))
        value = torch.from_numpy(np.array(values, copy=True)).reshape(tensor.shape)
        yield name, _restore_dense(name, value, config)


def iter_gguf_weights(
    model_path, device, *, include_moe_experts, include_non_moe, include_vision=True
):
    if include_moe_experts or not include_non_moe:
        raise ValueError("Clover GGUF requires native packed expert offload")
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    reader = _DenseReader(None, get_model_spec("CloverMoeModelLM"))
    for name, value in iter_hf_gguf_weights(model_path, config):
        merged = reader.add(name, value)
        if merged is None:
            yield name, value.to(device)
        else:
            for target, tensor in merged:
                yield target, tensor.to(device)
    if reader.missing():
        raise ValueError(
            f"Incomplete Clover GGUF projection groups: {reader.missing()}"
        )


def load_packed_expert_sources(model_path, config, *, layer_sink=None, dummy=False):
    from freetoken.moe.host_banks import (
        LayerCompletionTracker,
        PinPipeline,
        alloc_layer_banks,
    )

    if dummy:
        raise NotImplementedError(
            "Clover GGUF requires real packed weights; dummy banks are not defined"
        )
    _validate_experts(model_path, config)
    e, h, i = config.num_experts, config.hidden_size, config.moe_intermediate_size
    if h % 256 or i % 256:
        raise ValueError("Mixed Clover GGUF expert widths must be multiples of 256")
    hb = alloc_layer_banks(
        {
            "gate_up": ((e, 2 * i, h // 256 * 136), torch.uint8),
            "down": ((e, h, i // 256 * 144), torch.uint8),
        },
        config.num_layers,
    )
    banks = {
        name: [bank.tensor for bank in per_layer] for name, per_layer in hb.items()
    }

    def fill(sink):
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        for tensor in iter_gguf_tensors(model_path):
            match = _EXPERT.fullmatch(tensor.name)
            if match is None:
                continue
            layer, role = int(match[1]), match[2]
            if role == "down":
                destination = banks["down"][layer]
            else:
                begin = 0 if role == "gate" else i
                destination = banks["gate_up"][layer][:, begin : begin + i]
            destination.copy_(tensor.packed().reshape(destination.shape))
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        fill(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            fill(pins)
    else:
        fill(None)
    return banks
