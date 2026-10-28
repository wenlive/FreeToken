from types import SimpleNamespace

import torch

from freetoken.layers.moe import OffloadMoELayer


def test_offload_keeps_logical_experts_before_slot_rewrite():
    op = OffloadMoELayer.__new__(OffloadMoELayer)
    op.layer_id = 3
    op._lora_experts = {"active": True}
    logical = torch.tensor([[5, 2, 9]], dtype=torch.int32)
    slots = torch.tensor([[1, 6, 0]], dtype=torch.int32)
    cache = SimpleNamespace(is_cpu_layer=lambda _: False, decode_target="gpu",
                            ensure_experts=lambda _, ids: ids.copy_(slots), copy_missing=lambda: None,
                            bank_views=lambda: (), alphas_for_slots=lambda _: None)
    op.offload_cache = cache
    captured = {}
    def gemm(cache, x, weights, ids, **kwargs):
        captured.update(ids=ids.clone(), logical=kwargs["logical_ids"])
        return x
    op._expert_gemm = gemm
    ids = logical.clone()
    op._decode_routed(torch.ones(1, 4), torch.ones(1, 3), ids)
    assert torch.equal(captured["ids"], slots)
    assert torch.equal(captured["logical"], logical)
    assert captured["logical"].data_ptr() != ids.data_ptr()
