"""Real page/cache ownership: shared draft cells, fork boundaries and recycled pages."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.mtp_pool import MtpKVState, MTP_KV_BYTES_PER_TOKEN
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.models.qwen4_exp.mtp_state import MTP_LAST_RESIDUAL, capture_residual
from freetoken.scheduler.cache import CacheManager


def _pool():
    group = LinearGatedDeltaGroupConfig('linear', (0,), 2, 4, 16, 16, 4, 'silu')
    return LinearStatePool(group, 16, torch.float32, torch.device('cpu'), tp_size=1,
        slot_states=(SlotStateSpec(MTP_LAST_RESIDUAL, (10240,), dtype=torch.float32),), reserved_slots=4)


def test_real_prefix_forks_keep_shared_cells_and_isolate_lookahead():
    aux = MtpKVState(64, torch.device('cpu'))
    pool = _pool()
    pt = torch.full((4, 128), -1, dtype=torch.int32)
    manager = CacheManager(16, 4, pt, 'hybrid_radix', linear_state_pool=pool, mtp_state=aux)
    ids = [11, 12, 13, 14]

    def admit(tail, table_idx):
        tokens = torch.tensor([*ids, tail], dtype=torch.int32)
        pending = SimpleNamespace(input_ids=tokens, input_len=5)
        match = manager.match_req(pending)
        req = Req(tokens, table_idx, match.cuda_handle.cached_len, 4, table_idx,
                  SamplingParams(), match.cuda_handle)
        req.linear_slot_idx = pool.alloc(1)[0]
        req.mamba_ping_pong = tuple(pool.alloc(2))
        manager.lock(req.cache_handle)
        if req.cached_len:
            pt[table_idx, :req.cached_len].copy_(req.cache_handle.get_matched_indices())
            pool.copy_from(match.mamba_value, req.linear_slot_idx)
        manager.allocate_paged([req])
        return req

    cold = admit(15, 0)
    shared_keys = torch.arange(3 * 512, dtype=torch.float32).reshape(3, 2, 256)
    aux.publish(pt[0, 1:4], shared_keys, -shared_keys)
    pool.slot_state(MTP_LAST_RESIDUAL)[cold.linear_slot_idx].fill_(3.5)
    cold.cached_len = 4
    manager.cache_req(cold, finished=True)

    left, right = admit(25, 1), admit(35, 2)
    private_k, private_v = torch.zeros(8, 2, 256), torch.zeros(8, 2, 256)
    assert aux.restore_prefix(pt[1, :4], private_k, private_v) == 3
    torch.testing.assert_close(private_k[:3], shared_keys, rtol=0, atol=0)
    assert private_k[3].count_nonzero() == 0  # branch-dependent cell is not restored
    torch.testing.assert_close(pool.slot_state(MTP_LAST_RESIDUAL)[left.linear_slot_idx],
                               torch.full((10240,), 3.5), rtol=0, atol=0)
    assert torch.equal(pt[1, :4], pt[2, :4])
    assert pt[1, 4] != pt[2, 4]
    aux.publish(pt[1, 4:5], torch.full((1, 2, 256), 25.), torch.full((1, 2, 256), -25.))
    aux.publish(pt[2, 4:5], torch.full((1, 2, 256), 35.), torch.full((1, 2, 256), -35.))
    aux.restore_prefix(pt[1, :5], private_k, private_v)
    assert private_k[3, 0, 0] == 25
    aux.restore_prefix(pt[2, :5], private_k, private_v)
    assert private_k[3, 0, 0] == 35
    left.cached_len = right.cached_len = 5
    manager.cache_req(left, finished=True)
    manager.cache_req(right, finished=True)
    manager.check_integrity()

    # Re-admission reuses a physical page. Old auxiliary bytes may remain there,
    # but the allocator invalidates them before they can be restored as a prefix.
    manager.free_slots = manager.free_slots.sort().values
    again = admit(45, 1)
    with pytest.raises(RuntimeError, match='unpublished'):
        aux.restore_prefix(pt[1, :5], private_k, private_v)
    aux.restore_prefix(pt[1, :4], private_k, private_v)
    again.cached_len = 5
    manager.cache_req(again, finished=True)
    manager.check_integrity()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_auxiliary_rebuild_and_exact_owned_bytes(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    state = MtpKVState(64, torch.device(device))
    assert state.owned_bytes == 64 * MTP_KV_BYTES_PER_TOKEN
    slots = torch.tensor([3, 12], device=device)
    keys = torch.randn(2, 2, 256, device=device)
    state.publish(slots, keys, -keys)
    private_k, private_v = torch.zeros(8, 2, 256, device=device), torch.zeros(8, 2, 256, device=device)
    assert state.restore_prefix(torch.cat([slots[:1], slots]), private_k, private_v) == 2
    torch.testing.assert_close(private_k[:2], keys, rtol=0, atol=0)
    state.rebuild(128)
    assert state.owned_bytes == 128 * MTP_KV_BYTES_PER_TOKEN
    with pytest.raises(RuntimeError, match='unpublished'):
        state.restore_prefix(torch.cat([slots[:1], slots]), private_k, private_v)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_last_residual_tracks_every_row_and_follows_cow_rebuild(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from freetoken.attention.linear import FLAMetadata
    group = LinearGatedDeltaGroupConfig('linear', (0,), 2, 4, 16, 16, 4, 'silu')
    pool = LinearStatePool(group, 10, torch.float32, torch.device(device), tp_size=1,
        slot_states=(SlotStateSpec(MTP_LAST_RESIDUAL, (10240,), dtype=torch.float32),), reserved_slots=4)
    ids = torch.tensor([1], device=device, dtype=torch.int32)
    dst = torch.tensor(pool.reserved_slot_ids, device=device)
    boundaries = torch.arange(1, 5, device=device)
    fla = FLAMetadata(torch.tensor([0, 4], device=device), ids,
                      track_dst=dst, track_boundary_row=boundaries, is_verify=True)
    batch = SimpleNamespace(fla_metadata=fla)
    residual = torch.randn(4, 10256, device=device)[:, :10240]
    assert not residual.is_contiguous()
    buffer = torch.empty(8, 10240, device=device)
    capture_residual(residual, batch, pool, buffer)
    state = pool.slot_state(MTP_LAST_RESIDUAL)
    torch.testing.assert_close(buffer[:4], residual, rtol=0, atol=0)
    torch.testing.assert_close(state[1], residual[-1], rtol=0, atol=0)
    torch.testing.assert_close(state[dst], residual, rtol=0, atol=0)
    pool.copy_from(pool.reserved_slot_ids[1], 2)
    torch.testing.assert_close(state[2], residual[1], rtol=0, atol=0)
    pool.rebuild(13)
    assert pool.slot_state(MTP_LAST_RESIDUAL).count_nonzero() == 0
    assert pool.num_free_slots == 8
    assert set(pool.alloc(8)).isdisjoint(pool.reserved_slot_ids)
    pool.reclaim_all_slots()
    assert pool.num_free_slots == 8
    with pytest.raises(ValueError, match='Reserved'):
        pool.free(pool.reserved_slot_ids[0])
