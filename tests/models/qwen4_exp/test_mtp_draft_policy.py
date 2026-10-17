"""Paired decisions must resist common context drift and preserve the fixed map."""
import base64
import zlib
import pytest
import torch
from freetoken.models.qwen4_exp.mtp_draft_policy import PairedPolicy
from freetoken.models.qwen4_exp import mtp_draft_vocab as vocab


@pytest.mark.parametrize('winner',['full-3','cjk-2',None])
def test_paired_policy_handles_common_cost_drift(winner):
    policy=PairedPolicy()
    for step in range(512):
        arm=policy.choose()
        speed=(100+step*.15)*(1.15 if arm==winner else 1)
        committed=int(arm[-1])
        policy.observe(arm,committed-1,committed/speed)
    assert policy.preferred==(winner or 'full-3')


def test_exploitation_outliers_do_not_contaminate_the_paired_estimate():
    """A slow exploitation interval must not make the other arm win without paired evidence."""
    policy = PairedPolicy()
    for _ in range(320):
        arm = policy.choose()
        tokens = int(arm[-1])
        # Full wins every local pair. Only its later exploitation samples stall.
        speed = (120 if arm == 'full-3' else 100) if policy.record_pair else 1
        policy.observe(arm, tokens - 1, tokens / speed)
    assert policy.preferred == 'full-3'
    assert policy.snapshot()['rolling_cycle_tps']['full-3'] < 100
    assert all(ratio < 1 for ratio in policy.snapshot()['paired_rate_ratios'])


def test_periodic_probe_can_recover_after_the_winner_changes():
    """The controller must resume balanced measurements after a long exploitation interval."""
    policy = PairedPolicy()
    for winner in ('full-3', 'cjk-2', 'full-3'):
        for _ in range(640):
            arm = policy.choose()
            tokens = int(arm[-1])
            policy.observe(arm, tokens - 1, tokens / (140 if arm == winner else 100))
        assert policy.preferred == winner


def test_vocab_rows_are_bounded_unique_and_keep_argmax_tie_order():
    ids=vocab.load('cpu')
    assert ids.dtype==torch.int64 and len(ids)==106299
    assert int(ids[0])>=0 and int(ids[-1])<248320
    assert bool((ids[1:]>ids[:-1]).all())


def test_corrupted_vocabulary_is_rejected_before_device_allocation(monkeypatch):
    raw=zlib.decompress(base64.b85decode(vocab.DATA))
    changed=bytearray(raw);changed[32]^=1
    monkeypatch.setattr(vocab,'DATA',base64.b85encode(zlib.compress(changed)))
    with pytest.raises(ValueError,match='identity changed'):
        vocab.load('cpu')
