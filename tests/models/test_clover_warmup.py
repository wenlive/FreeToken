"""Failed startup allocation must not free unowned page-table entries."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.clover_moe.warmup import warmup_block_graphs


def test_failed_warmup_allocation_releases_only_owned_state_and_table(monkeypatch):
    events = []
    for name in ("memory_allocated", "memory_reserved", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    pool = SimpleNamespace(alloc=lambda count: [3], free=lambda slots: events.append(("state", slots)))
    table = torch.full((1, 32), 99, dtype=torch.int32)
    engine = SimpleNamespace(config=SimpleNamespace(clover_graph_warmup_widths=[8]),
        max_seq_len=32, device="cpu", linear_state_pool=pool, page_table=table,
        moe_offload_cache=None)
    runtime = SimpleNamespace(engine=engine, _require_block_backend=lambda manager: None)
    handle = object()

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("page budget exhausted")

    manager = SimpleNamespace(is_hybrid=True, ensure_mamba_slots=lambda count: None,
        match_req=lambda *args, **kwargs: SimpleNamespace(cuda_handle=handle),
        lock=lambda value: events.append(("lock", value)),
        unlock=lambda value: events.append(("unlock", value)), allocate_paged=fail_allocation,
        cache_req=lambda *args, **kwargs: pytest.fail("freed unowned page entries"))
    tables = SimpleNamespace(available_size=1, allocate=lambda: 0,
        free=lambda index: events.append(("table", index)))
    with pytest.raises(RuntimeError, match="page budget exhausted"):
        warmup_block_graphs(runtime, manager, tables)
    assert events == [("lock", handle), ("unlock", handle), ("state", [3]), ("table", 0)]
    assert (table == 99).all()
