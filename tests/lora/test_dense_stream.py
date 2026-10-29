from types import SimpleNamespace as NS

import pytest
import torch

from freetoken.layers.base import BaseOP, StateLessOP
from freetoken.lora.dense_stream import DenseLayerStreamer, load_streamed_dense_weights
from freetoken.models.qwen3_5_moe.model import Qwen3_5ForCausalLM


class Projection(BaseOP):
    def __init__(self, weight):
        self.weight = weight.clone()

    def forward(self, value):
        return value @ self.weight.T


class Layer(BaseOP):
    def __init__(self, weight):
        self.mlp = Projection(weight)

    def forward(self, value):
        return self.mlp.forward(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_stream_slot_reuse_matches_resident_layers_bytewise():
    torch.manual_seed(7)
    weights = [torch.randn(64, 64).bfloat16() for _ in range(3)]
    layers = [Layer(weight) for weight in weights]
    shared = StateLessOP()
    shared._cache = torch.arange(32, device='cuda')
    cache_pointer = shared._cache.data_ptr()
    for layer in layers:
        layer.rotary = shared
    stream = DenseLayerStreamer(layers, 'cuda')
    initial_hashes = stream.hashes()
    value = torch.randn(9, 64, device='cuda', dtype=torch.bfloat16)
    for sequence in [[0, 1, 2], [2, 0, 2, 1], [0, 1, 2]]:
        actual, reference = value.clone(), value.clone()
        for index in sequence:
            actual = layers[index].forward(actual)
            reference = reference @ weights[index].cuda().T
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
    assert stream.hashes() == initial_hashes
    assert shared._cache.is_cuda and shared._cache.data_ptr() == cache_pointer
    inventory = stream.inventory()
    assert inventory['slot_count'] == 1 and inventory['packed_host_all_pinned']
    assert inventory['gpu_slot_bytes'] == 64 * 64 * 2 and inventory['copies'] == 10
    other = torch.cuda.Stream()
    with torch.cuda.stream(other), pytest.raises(RuntimeError, match='construction stream'):
        layers[0].forward(value)


def test_streaming_rejects_graphs_before_loading_any_weight():
    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    config = NS(model_config=NS(is_moe=False), tp_info=NS(size=1), use_dummy_weight=False,
                active_encoders=(), mtp_runtime_path=None, gguf_embedding_host=False,
                cuda_graph_bs=None, cuda_graph_max_bs=1)
    with pytest.raises(ValueError, match='cuda-graph-max-bs 0'):
        load_streamed_dense_weights(model, config, 'cuda', lambda *args, **kwargs: pytest.fail('loaded before validation'))
