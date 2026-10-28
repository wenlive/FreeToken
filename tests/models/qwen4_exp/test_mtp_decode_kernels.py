"""MTP 核心回归：owner 顺序、草稿可见范围、卷积快照及 graph 元数据。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.moe.q2_adaptive import AdaptiveQ2Owners


def test_owner_selection_preserves_ties_gain_floor_and_swap_limit():
    """使用可手工核对的热度与归属，覆盖平局顺序、阈值与交换数量限制。"""
    controller = object.__new__(AdaptiveQ2Owners)
    controller.max_swaps = 3
    controller.cache = SimpleNamespace(protected_payload=SimpleNamespace(host_indices=[
        [-1, -1, -1, 0, 1, 2], [-1, -1, -1, 0, 1, 2]]))
    freq = np.array([[0, 1, 2, 8, 8, 3.49], [0, 1, 2, 8, 8, 3.5]], dtype=np.float32)
    assert controller.candidates(freq) == [(8., 0, 3, 0), (8., 1, 3, 0), (7., 0, 4, 1)]
    controller.max_swaps = 20
    assert controller.candidates(freq)[-1] == (1.5, 1, 5, 2)
    assert len(controller.candidates(freq)) == 5
    controller.cache.protected_payload.host_indices = [[-1] * 6, [0] * 6]
    assert controller.candidates(freq) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_dense_draft_attention_graph_uses_current_cell_and_hides_future_kv():
    """重放同一张图时更换 query/cell/KV，使用独立 SDPA 参考并污染未来 KV。"""
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention
    torch.manual_seed(819)
    capacity = 8192
    q = torch.randn(1, 24, 256, device='cuda') * .4
    keys = torch.randn(capacity // 64, 64, 2, 256, device='cuda') * .3
    values = torch.randn_like(keys)
    columns = torch.arange(capacity, device='cuda', dtype=torch.int32)[None, :]
    table = torch.arange(capacity // 64, device='cuda', dtype=torch.int32)[None, :]
    mapping = torch.zeros(1, device='cuda', dtype=torch.int32)
    cell = torch.zeros(1, device='cuda', dtype=torch.int32)
    def attention():
        return qsa_sparse_paged_attention(q, keys, values, columns, table, mapping,
                                          num_stages=1, draft_positions=cell)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attention()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = attention()
    torch.cuda.current_stream().wait_stream(stream)
    for count in [1, 64, 127, 2048, capacity]:
        cell.fill_(count - 1)
        q.normal_(std=.4)
        keys.normal_(std=.3)
        values.normal_()
        graph.replay()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.reshape(1, 24, 1, 256),
            keys.reshape(capacity, 2, 256)[:count].permute(1, 0, 2)[None],
            values.reshape(capacity, 2, 256)[:count].permute(1, 0, 2)[None],
            enable_gqa=True).reshape_as(output)
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-6)
        before = output.clone()
        keys.reshape(capacity, 2, 256)[count:].fill_(float('nan'))
        values.reshape(capacity, 2, 256)[count:].fill_(float('nan'))
        graph.replay()
        torch.testing.assert_close(output, before, rtol=0, atol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('count', [1, 2, 3, 4])
@pytest.mark.parametrize('snapshots', [False, True])
@pytest.mark.parametrize('strided', [False, True])
def test_verify_conv_graph_matches_decode_and_every_prefix(count, snapshots, strided):
    """直接对照既有逐行 decode，核对所有状态并重放新的输入和 live slot。"""
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode
    from freetoken.kernel.triton.gguf_gdn_conv import verify_conv

    torch.manual_seed(812 + count)
    channels = 259  # 覆盖最后一个不足 128 通道的 program。
    step = 2 if strided else 1
    x = torch.randn(count, channels * step, device='cuda')[:, ::step]
    weight = torch.randn(channels, 4 * step, device='cuda')[:, ::step]
    state = torch.randn(8, channels, 3, device='cuda')
    slots = torch.zeros(1, device='cuda', dtype=torch.int32)
    dst = torch.tensor([5, 1, 7, 2][:count], device='cuda') if snapshots else None
    boundary = torch.arange(count, 0, -1, device='cuda') if snapshots else None

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        verify_conv(x, weight, state, slots, dst, boundary)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = verify_conv(x, weight, state, slots, dst, boundary)
    torch.cuda.current_stream().wait_stream(stream)

    for live_slot in [0, 6]:
        x.normal_()
        weight.normal_(std=.3)
        state.normal_()
        slots.fill_(live_slot)
        expected_state = state.clone()
        rows = []
        for row in range(count):
            rows.append(causal_conv1d_decode(
                x[row:row + 1].clone(), expected_state, weight, slots))
            if snapshots:
                # boundary 逆序，验证 kernel 按实际前缀而非 track 下标保存。
                expected_state[dst[count - row - 1]].copy_(expected_state[live_slot])
        graph.replay()
        torch.testing.assert_close(output, torch.cat(rows), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_verify_packet_changes_request_slot_and_page_across_graph_replays(count):
    """真实 pinned H2D 后由图读取元数据，CPU 读回完成后才复用输入包。"""
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.kernel.triton.mtp_metadata import stage_verify_metadata

    capacity, page_size = 256, 64
    host = alloc_pinned_tensor(16, dtype=torch.int32)
    view = host.numpy()
    view.fill(0)
    packet = torch.zeros(16, device='cuda', dtype=torch.int32)
    pages = torch.randperm(8, device='cuda', dtype=torch.int32).reshape(2, 4)
    offsets = torch.arange(page_size, device='cuda', dtype=torch.int32)
    page_table = (pages[:, :, None] * page_size + offsets).reshape(2, capacity)
    page_table_cpu = page_table.cpu()
    metadata = SimpleNamespace(ring_slots=torch.empty(1, device='cuda', dtype=torch.int32),
        seq_lens=torch.empty(1, device='cuda', dtype=torch.int32),
        block_table=torch.empty(1, 4, device='cuda', dtype=torch.int32))
    batch = SimpleNamespace(input_ids=torch.empty(count, device='cuda', dtype=torch.int32),
        positions=torch.empty(count, device='cuda', dtype=torch.int32),
        out_loc=torch.empty(count, device='cuda', dtype=torch.int32),
        linear_table_idx=torch.empty(1, device='cuda', dtype=torch.int32),
        fla_metadata=SimpleNamespace(cache_indices=torch.empty(1, device='cuda', dtype=torch.int32)),
        attn_metadata=metadata)
    tensors = [batch.input_ids, batch.positions, batch.out_loc, batch.linear_table_idx,
        batch.fla_metadata.cache_indices, metadata.ring_slots, metadata.seq_lens, metadata.block_table]
    addresses = [value.data_ptr() for value in tensors]

    def stage():
        stage_verify_metadata(packet, page_table, batch, page_size)

    def consume():
        return torch.cat([value.reshape(-1) for value in tensors])

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stage()
        consume()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            observed = consume()
    torch.cuda.current_stream().wait_stream(stream)
    for owner, slot, position in [(0, 2, 1), (1, 6, 63), (0, 4, 127)]:
        tokens = list(range(100 + owner, 100 + owner + count))
        view[:4] = [position, owner, slot, count]
        view[4:4 + count] = tokens
        packet.copy_(host, non_blocking=True)
        stage()
        graph.replay()
        expected = [*tokens, *range(position, position + count),
            *page_table_cpu[owner, position:position + count].tolist(),
            slot, slot, owner, position + count,
            *(page_table_cpu[owner, ::page_size] // page_size).tolist()]
        assert observed.cpu().tolist() == expected
        assert [value.data_ptr() for value in tensors] == addresses
    graph.reset()
