"""Stop recognition must agree with streamed text and speculative clipping."""
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from freetoken.core import Req, SamplingParams
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.scheduler import Scheduler
from freetoken.tokenizer.detokenize import DetokenizeManager


@pytest.fixture
def tokenizer():
    """Use a real byte-level decoder with one merged ASCII token."""
    pieces = sorted(pre_tokenizers.ByteLevel.alphabet())
    pieces += ['pr', 'pre', 'pref', 'prefi', 'prefix']
    backend = Tokenizer(models.BPE(
        vocab={piece: i for i, piece in enumerate(pieces)},
        merges=[('p', 'r'), ('pr', 'e'), ('pre', 'f'), ('pref', 'i'), ('prefi', 'x')],
    ))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize('output,stops,expected', [
    ('ok 🦄', ['🦄'], 'ok '),
    ('ok 🦄🦄', ['🦄🦄'], 'ok '),
    ('ok 𠮷𠮷', ['𠮷𠮷'], 'ok '),
    ('prefix', ['fix', 'pre'], ''),
    ('prefix', ['pre', 'prefix'], ''),
    ('ok STOP', ['STOP'], 'ok '),
])
def test_native_stream_and_mtp_clip_at_the_same_stop(tokenizer, output, stops, expected):
    """Exercise real request history, byte decoding, clipping and stream holdback."""
    prompt = tokenizer.encode('prompt contains ' + ' '.join(stops), add_special_tokens=False)
    generated = tokenizer.encode(output, add_special_tokens=False)
    req = Req(torch.tensor(prompt, dtype=torch.int32), table_idx=0, cached_len=0,
              output_len=len(generated) + 4, uid=1,
              sampling_params=SamplingParams(stop_strs=stops), cache_handle=None)
    scheduler = SimpleNamespace(tokenizer=tokenizer, eos_token_ids=set())
    assert Scheduler._match_stop_str(scheduler, req) is None
    limit = Scheduler._mtp_delivery_limit(scheduler, req, generated)
    assert limit.finished
    assert req.input_ids.tolist() == prompt

    detokenizer = DetokenizeManager(tokenizer, frozenset())
    chunks = []
    for i, token in enumerate(generated):
        req.complete_one()
        req.append_host(torch.tensor([token], dtype=torch.int32))
        stop = Scheduler._match_stop_str(scheduler, req)
        chunks.extend(detokenizer.detokenize([DetokenizeMsg(
            uid=1, next_token=token, finished=stop is not None,
            finish_reason='stop' if stop else None, matched_stop=stop, stop_strs=stops,
        )]))
        if stop is not None:
            assert i + 1 == limit.count
            break
    assert ''.join(chunks) == expected
    assert detokenizer.decode_map == {}


def test_partial_unicode_stop_is_released_at_length(tokenizer):
    """An unfinished stop prefix remains visible when length ends generation."""
    tokens = tokenizer.encode('ok 🦄', add_special_tokens=False)
    manager = DetokenizeManager(tokenizer, frozenset())
    chunks = []
    for i, token in enumerate(tokens):
        finished = i == len(tokens) - 1
        chunks.extend(manager.detokenize([DetokenizeMsg(
            uid=2, next_token=token, finished=finished,
            finish_reason='length' if finished else None, stop_strs=['🦄🦄'],
        )]))
    assert ''.join(chunks) == 'ok 🦄'
    assert manager.decode_map == {}
