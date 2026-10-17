"""Check real CUDA loan lifetimes, retained metadata, and failure quarantine."""
from types import SimpleNamespace
import gc

import pytest
import torch

from freetoken.moe.prefill_workspace import PrefillWorkspace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def _fixture():
    """Use small real GPU banks with the production disposable-row boundaries."""
    banks = {role: torch.full((2064, width), 177, device='cuda', dtype=torch.uint8)
             for role, width in [('gate_up', 4096), ('down', 2048)]}
    for bank in banks.values():
        bank[2048:].fill_(231)
    ids = torch.full((2064,), -1, device='cuda', dtype=torch.int32)
    ids[1024:2048] = torch.arange(1024, device='cuda', dtype=torch.int32)
    mapping = torch.arange(1024, device='cuda', dtype=torch.int32) + 1024
    cache = SimpleNamespace(num_experts=512, bank_schema=tuple(banks),
        protected_payload=SimpleNamespace(dynamic=1024, protected_begin=2048, caches=banks),
        decode_copy_stream=torch.cuda.Stream(), prefill_copy_stream=torch.cuda.Stream(),
        id_of_slot=ids, slot_for_id=mapping,
        usage=torch.full((2064,), 19, device='cuda', dtype=torch.int64))
    batch = SimpleNamespace(input_ids=list(range(128)), attn_metadata=SimpleNamespace())
    return PrefillWorkspace(cache, torch.device('cuda')), batch, banks


def test_real_prefill_loan_preserves_metadata_and_bank_boundaries():
    """The original escaped row plans must survive without retaining a loan segment."""
    workspace, batch, banks = _fixture()
    for _ in range(2):
        def forward():
            batch.attn_metadata.cmp_rows = torch.arange(128, device='cuda', dtype=torch.int32)
            batch.attn_metadata.ring_rows = batch.attn_metadata.cmp_rows + 3
            return torch.arange(8192, device='cuda', dtype=torch.float32) * 3 + 7

        result = workspace.forward(forward, batch)
        assert torch.equal(result, torch.arange(8192, device='cuda', dtype=torch.float32) * 3 + 7)
        assert torch.equal(batch.attn_metadata.ring_rows,
                           torch.arange(128, device='cuda', dtype=torch.int32) + 3)
        record = workspace.records[-1]
        assert record['after_release'][8] == record['after_release'][7] == 0
        assert record['before_release'][1] > 0 and record['after_release'][2] == 0
        assert set(record['preserved_metadata']) == {'cmp_rows', 'ring_rows'}
        for bank in banks.values():
            assert bool((bank[:1024] == 177).all())
            assert bool((bank[2048:] == 231).all())
        assert bool((workspace.cache.slot_for_id == -1).all())
        assert bool((workspace.cache.id_of_slot[1024:2048] == -1).all())


def test_prefill_exception_keeps_backing_banks_and_blocks_future_execution():
    """An exception traceback can retain tensors, so the engine must require unload."""
    workspace, batch, _ = _fixture()

    def fail():
        batch.attn_metadata.cmp_rows = torch.arange(8192, device='cuda')
        raise ValueError('deliberate prefill failure')

    with pytest.raises(ValueError, match='deliberate'):
        workspace.forward(fail, batch)
    with pytest.raises(RuntimeError, match='unload'):
        workspace.check_usable()
    from freetoken.engine.engine import Engine
    from freetoken.models.qwen4_exp.mtp_runtime import MtpRuntime
    engine = Engine.__new__(Engine)
    engine.stream = torch.cuda.current_stream()
    engine.prefill_workspace = workspace
    runtime = MtpRuntime.__new__(MtpRuntime)
    runtime.engine = engine
    with pytest.raises(RuntimeError, match='unload'):
        engine.forward_batch(None, None)
    with pytest.raises(RuntimeError, match='unload'):
        engine.rebuild_runtime_cache()
    with pytest.raises(RuntimeError, match='unload'):
        runtime.forward(None, None, None, None, None, None)
    assert workspace._failed_regions and workspace._failed_pool
    assert workspace.statistics()[8] > 0
    batch.attn_metadata.cmp_rows = None
    workspace._failed_pool = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert workspace.statistics()[8] == 0


def test_unrecognized_escaped_tensor_cannot_be_reused_for_decode():
    """Keep the strict native live-block guard for survivors outside batch metadata."""
    workspace, batch, _ = _fixture()
    escaped = []

    def forward():
        escaped.append(torch.arange(8192, device='cuda'))
        return torch.ones(8, device='cuda')

    with pytest.raises(RuntimeError, match='did not return'):
        workspace.forward(forward, batch)
    with pytest.raises(RuntimeError, match='unload'):
        workspace.check_usable()
    assert workspace._failed_regions
    escaped.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert workspace.statistics()[8] == 0
