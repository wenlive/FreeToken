"""Union admission remaps original routes through actual GPU LRU evictions."""

import pytest
import torch

from freetoken.models.clover_moe.moe import admit_expert_union
from freetoken.moe.offload_cache import OffloadMoeCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 8, 32, 48, 80, 128])
def test_union_routes_survive_eviction_without_admitting_padding(monkeypatch, rows):
    cache = OffloadMoeCache(
        num_layers=3,
        num_experts=256,
        cache_size=512,
        device=torch.device("cuda"),
        prefill_overlap=False,
    )
    # This test owns metadata only. Byte movement is unchanged and separately
    # covered by the existing expert-bank tests and whole-model A/B run.
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    torch.manual_seed(rows)
    for layer in (0, 1, 2, 0):
        raw = torch.randint(0, 256, (rows, 8), device="cuda", dtype=torch.int32)
        original = raw.clone()
        slots = admit_expert_union(cache, layer, raw)
        assert torch.equal(raw, original)
        assert torch.equal(cache.id_of_slot[slots.long()], layer * 256 + raw)
        fetched = set(cache.src_indices[: int(cache.num_indices.item())].cpu().tolist())
        assert fetched <= set(raw.flatten().cpu().tolist())
