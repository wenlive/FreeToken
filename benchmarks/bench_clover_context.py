"""Warm preview/verify latency at exact long prefixes, with capture excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--cache-slots', type=int, default=2304)
    p.add_argument('--prefixes', default='1024,4096,16384,32768')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    prefixes = [int(x) for x in a.prefixes.split(',')]
    llm = LLM(a.model, moe_strategy='offload', moe_cache_size=a.cache_slots,
        expert_load='serial', max_running_req=1, memory_ratio=.95,
        num_token_override=max(prefixes)+128, max_seq_len_override=max(prefixes)+128,
        max_extend_tokens=4096, cuda_graph_bs=[1], clover_max_width=32)
    engine, runtime = llm.engine, llm.engine.mtp_runtime
    match = llm.cache_manager.match_req
    llm.cache_manager.match_req = lambda req, max_len=None: match(req, max_len=0)
    result = dict(state='running', scope='Exact forced prefix; CUDA event call envelopes; '
        'one capture excluded then two warmups and five measurements; resident expert cache',
        cache_slots=a.cache_slots, cases=[])
    current = None
    original = engine.forward_batch

    def save():
        (a.output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    def audited(batch, sampling):
        output = original(batch, sampling)
        req = batch.reqs[0]
        if batch.is_prefill and req.cached_len == current:
            runtime._blocks().prepare_clean(req)
            llm.cache_manager.allocate_paged([req], reserve_tokens=31)
            for width in (8, 16, 32):
                tokens = torch.full((width,), runtime.mask_id, device=engine.device, dtype=torch.int32)
                for kind, operation in (
                    ('preview', lambda: runtime._preview(req, tokens)),
                    ('verify', lambda: runtime._verify(req, tokens, width)),
                ):
                    begin = time.perf_counter()
                    operation()
                    torch.cuda.synchronize()
                    capture_wall = time.perf_counter()-begin
                    timings = []
                    for _ in range(7):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(engine.stream)
                        operation()
                        end.record(engine.stream)
                        end.synchronize()
                        timings.append(start.elapsed_time(end))
                    row = dict(prefix=current, width=width, kind=kind,
                        first_call_wall_seconds=capture_wall, warmup_ms=timings[:2],
                        measured_ms=timings[2:], median_ms=statistics.median(timings[2:]))
                    result['cases'].append(row)
                    print(json.dumps(row), flush=True)
                    save()
        return output

    engine.forward_batch = audited
    try:
        seed = llm.tokenizer.apply_chat_template([dict(role='user', content='Describe a tree.')],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        pattern = llm.tokenizer.encode('Trees shade the streets. ')
        for current in prefixes:
            middle = current-len(seed)
            ids = seed[:-8] + (pattern*((middle+len(pattern)-1)//len(pattern)))[:middle] + seed[-8:]
            assert len(ids) == current
            result.setdefault('inputs', []).append(dict(prefix=current, token_ids=ids,
                sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
            llm.generate([ids], SamplingParams(temperature=0, max_tokens=64, clover_mode='causal'))
        result.update(state='complete', cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved=torch.cuda.max_memory_reserved())
        save()
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
