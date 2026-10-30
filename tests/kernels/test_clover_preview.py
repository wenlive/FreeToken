"""Clover operators against fixed official HF functions and native causal state."""
import os
from pathlib import Path
import runpy

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def official():
    source = os.environ.get("CLOVER_HF_SOURCE")
    if not source:
        pytest.skip("set CLOVER_HF_SOURCE to the existing fixed official HF source")
    harness = runpy.run_path(str(Path(__file__).parents[2] / "benchmarks/clover_hf_reference.py"))
    return harness["official_modules"](Path(source))[0]


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("width", [8, 16, 32])
def test_preview_convolution_matches_official_and_does_not_write_history(official, width):
    from freetoken.kernel.triton.clover_conv import clover_local_memory_conv
    torch.manual_seed(width)
    channels, batch = 256, 2
    x = torch.randn(batch * width, channels, device="cuda", dtype=torch.bfloat16)
    history = torch.randn(batch, channels, 3, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(channels, 1, 4, device="cuda", dtype=torch.bfloat16) * .1
    right = torch.randn(channels, 1, 3, device="cuda", dtype=torch.bfloat16) * .1
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    valid = torch.ones(batch * width, device="cuda", dtype=torch.bool)
    valid[-2:] = False
    frozen = history.clone()
    options = dict(clean_conv_states=history, cache_indices=indices, conv_weights=weights,
                   block_size=width, active_valid_mask=valid, conv_right_weight=right)
    expected = official.bd_gdn_conv1d(x, **options)
    actual = clover_local_memory_conv(x, bias=None, activation="silu", conv_mode="independent_symmetric", **options)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=.02, atol=.004)
    assert torch.equal(history, frozen)


@pytest.mark.parametrize("width", [8, 16, 32])
def test_preview_gdn_matches_official_and_does_not_write_state(official, width):
    from freetoken.kernel.triton.clover_gdn import clover_local_memory_triton
    torch.manual_seed(width)
    batch, kh, vh, dim = 2, 4, 8, 128
    q, k = [torch.randn(1, batch * width, kh, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(1, batch * width, vh, dim, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, batch * width, vh, device="cuda")
    state = torch.randn(batch, vh, dim, dim, device="cuda") * .1
    frozen = state.clone()
    valid = torch.ones(batch * width, device="cuda", dtype=torch.bool)
    valid[-2:] = False
    options = dict(clean_ssm_states=state, cache_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
                   active_valid_mask=valid, scale=dim ** -.5,
                   softmax_logit_scale=torch.full((vh,), 10., device="cuda"),
                   softmax_history_bias=torch.full((vh,), 5., device="cuda"),
                   softmax_history_c=torch.ones(vh, device="cuda"))
    expected = official.bd_gdn_local_memory(q, k, v, g, **options)
    actual = clover_local_memory_triton(q, k, v, g, **options)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=.02, atol=.006)
    assert torch.equal(state, frozen)


@pytest.mark.parametrize("width", [8, 16, 32])
def test_coreset_matches_official_stable_topk(official, width):
    from freetoken.kernel.triton.clover_coreset import clover_block_coreset_triton
    torch.manual_seed(width)
    logits = torch.randn(2 * width, 256, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, width, device="cuda", dtype=torch.bool)
    expected = official.coreset_torch(logits.reshape(2, width, 256), valid, 8, .9).reshape_as(logits)
    actual = clover_block_coreset_triton(logits, width, 8, .9)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width,prefix", [(8, 0), (16, 55), (32, 129)])
def test_split_attention_is_bidirectional_and_read_only(width, prefix):
    from freetoken.kernel.triton.attention import extend_paged_attention
    torch.manual_seed(width)
    q = torch.randn(width, 16, 256, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(max(1, prefix) + 16, 2, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    current_k, current_v = [torch.randn(width, 2, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    slots = torch.randperm(k.shape[0], device="cuda")[:prefix].int()
    frozen_k, frozen_v = k.clone(), v.clone()
    actual = extend_paged_attention(q, k, v, torch.tensor([0, width], device="cuda", dtype=torch.int32),
        torch.tensor([0, prefix + width], device="cuda", dtype=torch.int32), slots,
        torch.tensor([prefix], device="cuda", dtype=torch.int32), width, 256 ** -.5,
        k_extend=current_k, v_extend=current_v,
        block_ends=torch.full((width,), prefix+width, device="cuda", dtype=torch.int32))
    keys, values = [torch.cat([old[slots.long()], new]).repeat_interleave(8, 1).transpose(0, 1).float() for old, new in ((k,current_k),(v,current_v))]
    expected = F.scaled_dot_product_attention(q.transpose(0, 1).float(), keys, values).transpose(0, 1)
    torch.testing.assert_close(actual.float(), expected, rtol=.02, atol=.006)
    assert torch.equal(k, frozen_k) and torch.equal(v, frozen_v)


@pytest.mark.parametrize("width", [8, 16, 32])
def test_existing_fla_verify_snapshots_equal_sequential_decode(width):
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as update
    torch.manual_seed(width)
    kh, vh, dim = 4, 8, 128
    q, k = [torch.randn(1,width,kh,dim,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(1,width,vh,dim,device="cuda",dtype=torch.bfloat16)
    a, b = [torch.randn(width,vh,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    state = torch.randn(1,vh,dim,dim,device="cuda") * .1
    frozen, sequential = state.clone(), state.clone()
    snapshots = torch.empty(1,width,vh,dim,dim,device="cuda")
    index = torch.zeros(1,device="cuda",dtype=torch.int32)
    options = dict(A_log=torch.randn(vh,device="cuda"), dt_bias=torch.randn(vh,device="cuda"),
                   softplus_beta=1., softplus_threshold=20., initial_state_indices=index,
                   scale=dim ** -.5, use_qk_l2norm_in_kernel=True)
    actual = update(q=q,k=k,v=v,a=a,b=b,initial_state_source=state, disable_state_update=True,
                    intermediate_states_buffer=snapshots, intermediate_state_indices=index,
                    cu_seqlens=torch.tensor([0,width],device="cuda",dtype=torch.int32), **options)
    expected = []
    for row in range(width):
        expected.append(update(q=q[:,row:row+1],k=k[:,row:row+1],v=v[:,row:row+1],a=a[row:row+1],b=b[row:row+1],
            initial_state_source=sequential, cu_seqlens=torch.tensor([0,1],device="cuda",dtype=torch.int32), **options))
        torch.testing.assert_close(snapshots[:,row], sequential, rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.cat(expected,dim=1), rtol=0, atol=0)
    assert torch.equal(state, frozen)
