"""Owned page limits and verified output boundaries for speculative windows."""
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.models.qwen4_exp.mtp_runtime import verified_prefix, window_depth
from freetoken.scheduler.cache import CacheManager


def test_window_never_requires_an_unavailable_page():
    table = torch.zeros((2, 128), dtype=torch.int32)
    manager = CacheManager(1, 64, table, 'hybrid_radix')
    ids = torch.arange(62, dtype=torch.int32)
    handle = manager.match_req(SimpleNamespace(input_ids=ids, input_len=len(ids))).cuda_handle
    req = Req(ids, 0, 0, 17, 1, SamplingParams(max_tokens=17), handle)
    manager.allocate_paged([req])
    req.cached_len = 62
    req.device_len = 63
    assert manager.available_size == 0
    assert window_depth(req, manager, 8192) == 1
    manager.allocate_paged([req], reserve_tokens=1)
    assert req.mtp_reserved_len == 64
    req.cached_len, req.device_len = 63, 64
    assert window_depth(req, manager, 8192) == 0
    manager.allocate_paged([req], reserve_tokens=0)
    assert req.mtp_reserved_len == 64


def test_window_can_use_an_existing_lookahead_page():
    table = torch.zeros((2, 128), dtype=torch.int32)
    manager = CacheManager(2, 64, table, 'hybrid_radix')
    ids = torch.arange(63, dtype=torch.int32)
    handle = manager.match_req(SimpleNamespace(input_ids=ids, input_len=len(ids))).cuda_handle
    req = Req(ids, 0, 0, 17, 1, SamplingParams(max_tokens=17), handle)
    manager.allocate_paged([req], reserve_tokens=3)
    assert manager.available_size == 0 and req.mtp_reserved_len == 128
    req.cached_len, req.device_len = 63, 64
    assert window_depth(req, manager, 8192) == 3
    assert window_depth(req, manager, 64) == 1
    req.max_device_len = 65
    assert window_depth(req, manager, 8192) == 0


def test_eos_never_accepts_a_draft_tail_after_termination():
    assert verified_prefix([7, 2, 9], [7, 2, 9, 10], {2}, False) == [7, 2]
    assert verified_prefix([7, 2, 9], [7, 2, 9, 10], {2}, True) == [7, 2, 9, 10]
    assert verified_prefix([7, 8, 9], [7, 3, 9, 10], {2}, False) == [7, 3]
    assert verified_prefix([], [4], {2}, False) == [4]


def test_drafter_checks_the_bytes_it_actually_loads(tmp_path, monkeypatch):
    """Replace a real tensor file at the read boundary; unchecked second-read bytes must never load."""
    from freetoken.models.qwen4_exp.mtp_drafter import MtpDrafter

    original = torch.tensor([1.25], dtype=torch.float32).numpy().tobytes()
    changed = torch.tensor([9.5], dtype=torch.float32).numpy().tobytes()
    rows = []
    for index in range(31):
        path = tmp_path / f'tensor-{index}.bin'
        path.write_bytes(original)
        rows.append(dict(name=f'parameter-{index}', path=path.name, type_id=0,
                         shape=[1], sha256=hashlib.sha256(original).hexdigest()))
    (tmp_path/'runtime.json').write_text(json.dumps(dict(status='packed_unvalidated_runtime', tensors=rows)))
    embedding = SimpleNamespace(packed_0=torch.empty(1, dtype=torch.uint8))
    main = SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding,
        layers=SimpleNamespace(op_list=[SimpleNamespace(_is_linear=False,
                                                        self_attn=SimpleNamespace(rotary=object()))])),
        lm_head=object(), _config=SimpleNamespace(norm_topk_prob=True))
    target = tmp_path / rows[0]['path']
    read_bytes = Path.read_bytes

    def replace_before_read(path):
        if path == target:
            target.write_bytes(changed)
        return read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', replace_before_read)
    with pytest.raises(ValueError, match='MTP runtime bytes changed'):
        MtpDrafter(tmp_path, main, max_context=64)
