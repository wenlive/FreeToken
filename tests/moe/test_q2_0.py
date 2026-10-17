"""External expert overrides must preserve the trunk's independent schemes."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.layers.quantization.configs.base import ExpertSourceConfig, NoQuantConfig
from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.q2_0 import Q20MoEMethod
from freetoken.layers.quantization.names import NameMap


def test_q2_override_preserves_trunk_and_shared_experts():
    quant = ExpertSourceConfig(NoQuantConfig(NameMap(roots=(("model.layers", "model.language_model.layers"),))))
    for name in ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.shared_expert.gate_up_proj",
                 "model.layers.0.mlp.shared_expert_gate", "model.layers.0.mlp.gate", "lm_head"):
        assert quant.scheme_for(name) is None
    scheme = quant.scheme_for("model.layers.0.mlp.experts")
    assert scheme.kind is QuantKind.Q2_0
    assert quant.scheme_for_name("model.language_model.layers.0.mlp.experts.511.down_proj") == scheme


def test_q2_layout_packing_preserves_expert_rows_without_expansion():
    cfg = MoEConfig(num_experts=3, hidden=64, intermediate=64, top_k=2, strategy="offload")
    method = Q20MoEMethod(cfg)
    layout = method.layout()
    assert sum(torch.empty(spec.shape, dtype=spec.dtype).numel() for spec in layout.values()) == 3 * 64 * 64 // 64 * 18
    pieces = {role: torch.randint(0, 256, (3, 64, 18), dtype=torch.uint8) for role in ("gate", "up", "down")}
    out = {role: torch.empty((3, *spec.shape), dtype=spec.dtype) for role, spec in layout.items()}
    method.pack(pieces, out)
    torch.testing.assert_close(out["gate_up"][:, :64], pieces["gate"], rtol=0, atol=0)
    torch.testing.assert_close(out["gate_up"][:, 64:], pieces["up"], rtol=0, atol=0)
    torch.testing.assert_close(out["down"], pieces["down"], rtol=0, atol=0)
    assert method.cpu_format is None
    for changes in ({"decode_target": "cpu"}, {"strategy": "resident"}, {"hidden": 65},
                    {"dtype": torch.float16}, {"apply_router_weight_on_input": True}):
        with pytest.raises(RuntimeError):
            Q20MoEMethod(replace(cfg, **changes))


def test_q2_source_is_explicit_and_architecture_scoped():
    config = SimpleNamespace(model_type="qwen4_exp", freetoken_expert_source={"format": "q2_0"})
    assert QuantConfig.from_hf(config).scheme_for("model.layers.0.mlp.experts").kind is QuantKind.Q2_0
    config.model_type = "qwen3_5_moe"
    with pytest.raises(ValueError):
        QuantConfig.from_hf(config)
