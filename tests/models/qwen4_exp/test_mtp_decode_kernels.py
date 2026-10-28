"""候选核心回归：稳定 owner 顺序、动态草稿可见范围和 graph 元数据地址。"""
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
def test_verify_packet_changes_request_slot_and_page_across_graph_replays(count):
    """请求 owner、SSM slot 和跨页位置都来自当前包，graph tensor 地址保持不变。"""
    from freetoken.kernel.triton.mtp_metadata import stage_verify_metadata
    capacity, page_size = 256, 64
    packet = torch.zeros(16, device='cuda', dtype=torch.int32)
    pages = torch.randperm(8, device='cuda', dtype=torch.int32).reshape(2, 4)
    offsets = torch.arange(page_size, device='cuda', dtype=torch.int32)
    page_table = (pages[:, :, None] * page_size + offsets).reshape(2, capacity)
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
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stage()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            stage()
    torch.cuda.current_stream().wait_stream(stream)
    for owner, slot, position in [(0, 2, 1), (1, 6, 63), (0, 4, 127)]:
        tokens = list(range(100 + owner, 100 + owner + count))
        packet[:4] = torch.tensor([position, owner, slot, count], device='cuda')
        packet[4:4 + count] = torch.tensor(tokens, device='cuda')
        graph.replay()
        assert batch.input_ids.cpu().tolist() == tokens
        assert batch.positions.cpu().tolist() == list(range(position, position + count))
        torch.testing.assert_close(batch.out_loc, page_table[owner, position:position + count])
        assert batch.linear_table_idx.item() == batch.fla_metadata.cache_indices.item() == slot
        assert metadata.ring_slots.item() == owner
        assert metadata.seq_lens.item() == position + count
        torch.testing.assert_close(metadata.block_table[0], page_table[owner, ::page_size] // page_size)
        assert [value.data_ptr() for value in tensors] == addresses
    graph.reset()
