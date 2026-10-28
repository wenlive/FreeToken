"""Offline text must obey scheduler stop metadata, including sub-token stops."""
import pytest
from tokenizers import Tokenizer, decoders, models

from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.message import DetokenizeMsg, PromptAdmittedMsg


class TokenReplyLLM(LLM):
    """Drive real offline admission/accounting with deterministic scheduler replies."""
    def __init__(self, replies):
        self.replies = replies
        self.prefill_budget = 1024
        self.tokenizer = Tokenizer(models.WordLevel({
            'prompt':0, 'x':1, 'abcSTOP':2, 'STOP':3, 'plain':4,
            '<eos>':5, 'prefix':6, '🦄🦄':7,
        }))
        self.tokenizer.decoder = decoders.Fuse()
        self.eos_token_ids = {5}

    def run_forever(self):
        for request in self.offline_receive_msg(False):
            tokens, matched = self.replies[request.uid]
            self.offline_send_result([PromptAdmittedMsg(uid=request.uid, prompt_tokens=len(request.input_ids))])
            for index, token in enumerate(tokens):
                finished = index == len(tokens)-1
                self.offline_send_result([DetokenizeMsg(uid=request.uid, next_token=token,
                    finished=finished, finish_reason='stop' if finished else None,
                    matched_stop=matched if finished else None)])


@pytest.mark.parametrize('tokens,matched,expected', [
    ([2], 'STOP', 'abc'),
    ([3], 'STOP', ''),
    ([6], 'pre', ''),
    ([7], '🦄🦄', ''),
    ([4,5], None, 'plain'),
    ([4], None, 'plain'),
])
def test_generate_projects_the_reported_stop_without_losing_consumed_ids(tokens, matched, expected):
    """The output may stop inside one consumed token; trim text while retaining its token accounting."""
    runtime = TokenReplyLLM([(tokens, matched)])
    result, = runtime.generate([[0,3]], SamplingParams(stop_strs=[matched] if matched else []))
    assert result['text'] == expected
    assert result['token_ids'] == [token for token in tokens if token != 5]


def test_each_offline_request_owns_its_stop_projection():
    """A stopped row must not trim a different row or the next generation call."""
    runtime = TokenReplyLLM([([2], 'STOP'), ([4], None)])
    results = runtime.generate([[0], [0]], SamplingParams(stop_strs=['STOP']))
    assert [row['text'] for row in results] == ['abc', 'plain']
    runtime.replies = [([4], None)]
    result, = runtime.generate([[0]], SamplingParams())
    assert result['text'] == 'plain'
