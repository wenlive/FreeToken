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
    handle = SimpleNamespace(cached_len=0)

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


@pytest.mark.parametrize("publish", [False, True])
def test_warmup_rejects_publishing_a_synthetic_prefix(monkeypatch, publish):
    from freetoken.models.clover_moe import warmup
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.table import TableManager
    from tests.scheduler.test_hybrid_cache_manager import _pool

    for name in ("memory_allocated", "memory_reserved", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(warmup, "_warmup_sampling", lambda *args: None)
    pool = _pool()
    page_table = torch.zeros(4, 32, dtype=torch.int32)
    manager = CacheManager(32, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    tables = TableManager(4, page_table)
    engine = SimpleNamespace(config=SimpleNamespace(clover_graph_warmup_widths=[8], clover_coreset_top_p=.9),
        max_seq_len=32, device="cpu", linear_state_pool=pool, page_table=page_table,
        moe_offload_cache=None, model=SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[]))))
    blocks = SimpleNamespace(prepare_clean=lambda req: None, clear_verify=lambda: None)
    runtime = SimpleNamespace(engine=engine, transactions=None, mask_id=16,
        _require_block_backend=lambda manager: None, _blocks=lambda: blocks,
        _preview=lambda req, tokens: torch.zeros(len(tokens), 17),
        _verify=lambda req, tokens, width: torch.zeros(width, 17))
    original = manager.cache_req

    def cache_req(req, *, finished):
        if publish:
            req.cached_len = 1
        original(req, finished=finished)

    monkeypatch.setattr(manager, "cache_req", cache_req)
    if publish:
        with pytest.raises(AssertionError, match="synthetic prefix"):
            warmup_block_graphs(runtime, manager, tables)
    else:
        warmup_block_graphs(runtime, manager, tables)
    manager.check_integrity()
    assert tables.available_size == 4
