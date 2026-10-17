"""Owned page limits and verified output boundaries for speculative windows."""
from types import SimpleNamespace

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
