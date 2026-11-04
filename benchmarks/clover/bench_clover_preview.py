"""Compare width-W native previews on the exact stock first-anchor inputs."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from benchmarks.clover.harness import PROMPTS, load_engine, save_results
from benchmarks.clover.check_clover_preview import persistent_hashes


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--cache-slots', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    llm = load_engine(args.model, factory=LLM, cache_slots=args.cache_slots,
        memory_ratio=.85, context=4096, clover_max_width=16,
        moe_collect_stats=True, clover_disable_penalties=True)
    engine = llm.engine
    runtime = engine.mtp_runtime
    forward = engine.forward_batch
    result = dict(state='running', scope='P0 exact stock prompt plus its greedy next-token anchor, masks in remaining W-1 rows',
        warmups=2, repeats=5, cases=[], causal_ms={})
    current = None
    def save():
        save_results(args.output, result)

    def audited(batch, sampling):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(engine.stream)
        output = forward(batch, sampling)
        end.record(engine.stream)
        end.synchronize()
        if batch.is_decode:
            result['causal_ms'][current].append(begin.elapsed_time(end))
        if batch.is_prefill:
            req = batch.reqs[0]
            frozen = persistent_hashes(engine)
            runtime._blocks().prepare_clean(req)
            for width in (8, 16):
                tokens = torch.full((width,), runtime.mask_id, device=engine.device, dtype=torch.int32)
                tokens[:1].copy_(output.next_tokens_gpu[:1])
                runtime._preview(req, tokens)  # capture excluded
                engine.moe_offload_cache.reset()
                timings = []
                for repeat in range(7):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(engine.stream)
                    runtime._preview(req, tokens)
                    end.record(engine.stream)
                    end.synchronize()
                    timings.append(begin.elapsed_time(end))
                after = persistent_hashes(engine)
                assert frozen == after, 'Preview modified persistent KV/GDN state'
                result['cases'].append(dict(prompt=current, width=width, prefix=req.cached_len,
                    anchor=int(tokens[0]), measured_ms=timings[2:], warmup_ms=timings[:2],
                    median_ms=statistics.median(timings[2:]), hashes_before=frozen, hashes_after=after))
                save()
        return output

    engine.forward_batch = audited
    try:
        for name, text in PROMPTS.items():
            current = name
            result['causal_ms'][name] = []
            ids = llm.tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            llm.generate([ids], SamplingParams(temperature=0, max_tokens=64, clover_mode='causal'))
        result['causal_median_ms'] = {name: statistics.median(values) for name, values in result['causal_ms'].items()}
        result['state'] = 'complete'
        save()
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
