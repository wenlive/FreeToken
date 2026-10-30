"""BF16 real-layer comparisons without loading the complete BF16 model."""
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import safetensors
import torch


pytestmark = [pytest.mark.needs_weights, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.fixture(scope="module", params=[0, 36])
def gdn_pair(request):
    location = os.environ.get("CLOVER_HF_SOURCE")
    if not location:
        pytest.skip("set CLOVER_HF_SOURCE to the fixed local checkpoint")
    source = Path(location)
    harness = runpy.run_path(str(Path(__file__).parents[2] / "benchmarks/clover_hf_reference.py"))
    official, configuration, _ = harness["official_modules"](source)
    config = configuration.CloverMoeConfig(**json.loads((source / "config.json").read_text(encoding="utf-8")))
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.quantization import finalize_quant
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
    from freetoken.models.qwen3_5_moe.weight import _DenseReader
    from freetoken.models.register import get_model_spec
    from freetoken.utils.torch_utils import torch_dtype
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = request.param
    prefix = f"model.layers.{layer}.linear_attn"
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        reference = official.CloverGatedDeltaNet(config)
        native = Qwen3_5GatedDeltaNet(config.hidden_size, config.linear_num_key_heads, config.linear_num_value_heads,
            config.linear_key_head_dim, config.linear_value_head_dim, config.linear_conv_kernel_dim,
            config.rms_norm_eps, layer, prefix=prefix)
        native.conv_right_weight = torch.empty(native.conv_dim, 1, 3)
        for name in ("bd_softmax_logit_scale", "bd_softmax_history_bias", "bd_softmax_history_c"):
            setattr(native, name, torch.empty(native.num_v_heads))
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    reader = _DenseReader(None, get_model_spec("CloverMoeModelLM"))
    loaded = {}
    for key in sorted(index):
        if not key.startswith(prefix + "."):
            continue
        with safetensors.safe_open(source / index[key], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key).to("cuda")
        relative = key[len(prefix) + 1:]
        parent, _, leaf = relative.rpartition(".")
        owner = reference.get_submodule(parent) if parent else reference
        setattr(owner, leaf, torch.nn.Parameter(tensor, requires_grad=False))
        values = reader.add(key, tensor)
        if values is None:
            loaded[key] = tensor
        else:
            loaded.update(values)
    expected = native.state_dict(prefix=prefix)
    assert set(loaded) == set(expected) and not reader.pending
    native.load_state_dict({key: value.to(expected[key].dtype) for key, value in loaded.items()}, prefix=prefix)
    finalize_quant(native)
    reference.eval()
    return native, reference, config, layer


@pytest.mark.parametrize("mode,width", [("decode", 1), ("prefill", 8), ("preview", 8), ("preview", 16), ("preview", 32), ("verify",8), ("verify",16)])
@torch.inference_mode()
def test_real_bf16_gdn_matches_official(gdn_pair, monkeypatch, mode, width):
    native, official, config, layer = gdn_pair
    from freetoken.attention.linear import FLAMetadata
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.clover_moe.preview import gdn_forward
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    import freetoken.core as core
    group = LinearGatedDeltaGroupConfig("linear", (layer,), native.num_k_heads, native.num_v_heads,
                                        native.head_k_dim, native.head_v_dim, 4, "silu")
    pool = LinearStatePool(group, 2, torch.bfloat16, torch.device("cuda"), tp_size=1)
    torch.manual_seed(width + layer)
    pool.conv_states[0, 1].normal_(std=.05)
    pool.recurrent_states[0, 1].normal_(std=.01)
    conv, state = pool.conv_states[0, 1:2].clone(), pool.recurrent_states[0, 1:2].clone()
    hidden = torch.randn(width, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, width, device="cuda", dtype=torch.bool)
    req = SimpleNamespace(linear_slot_idx=1, table_idx=1)
    batch = SimpleNamespace(reqs=[req], is_decode=mode == "decode", clover_valid=valid.flatten(),
        fla_metadata=FLAMetadata(torch.tensor([0,width],device="cuda",dtype=torch.int32),
            torch.ones(1,device="cuda",dtype=torch.int32), torch.ones(1,device="cuda",dtype=torch.bool)))
    ctx = SimpleNamespace(batch=batch, linear_state_pool=pool)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    if mode == "verify":
        from freetoken.models.clover_moe.preview import CloverTransactions
        batch.clover_transactions = CloverTransactions(pool,16)
        batch.clover_transactions.begin(width)
    reference, _ = official(hidden.unsqueeze(0), {"conv": conv.clone(), "ssm": state.clone()}, valid, mode == "preview")
    actual = gdn_forward(native, hidden, ctx, preview=mode=="preview") if mode in ("preview","verify") else native.forward(hidden)
    error = (actual.float() - reference[0].float()).norm() / reference.float().norm()
    print(json.dumps(dict(layer=layer, mode=mode, width=width, relative_l2=float(error),
                         max_abs=float((actual.float()-reference[0].float()).abs().max()))))
    assert float(error) < .01
    if mode in ("preview","verify"):
        assert torch.equal(pool.conv_states[0,1:2], conv)
        assert torch.equal(pool.recurrent_states[0,1:2], state)
