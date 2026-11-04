"""Measure descriptor traversal and reader construction without guessing payload I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import gguf
import freetoken.models.clover_moe.gguf as adapter
from freetoken.llm import LLM

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--hash-weights', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
records = []

def measure(owner, name):
    original = getattr(owner, name)
    def observed(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        records.append(dict(operation=name, seconds=time.perf_counter() - start))
        return result
    setattr(owner, name, observed)

measure(gguf, 'GGUFReader')
measure(adapter, '_validate_experts')
measure(adapter, 'parse_gguf_config')
begin = time.perf_counter()
llm = LLM(a.model, moe_strategy='offload', moe_cache_auto=True, memory_ratio=.9,
    kv_reserve_tokens=4096, num_token_override=4096, max_seq_len_override=4096,
    max_extend_tokens=4096, max_running_req=1, clover_max_width=0,
    expert_load='serial', cuda_graph_bs=[1])
try:
    result = dict(state='complete', startup_seconds=time.perf_counter() - begin, calls=records,
        scope='Nested timings overlap; do not sum them as disjoint costs. No payload scan inferred.')
    result['settings'] = dict(model=a.model, cache_slots=llm.engine.moe_offload_cache.cache_size,
        memory_ratio=.9, context=llm.engine.max_seq_len, clover_max_width=0,
        mapped_host_table_bytes=llm.engine._host_tables_bytes)
    if a.hash_weights:
        import torch
        hashes = {}
        def fingerprint(name, value):
            host = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
            hashes[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(memoryview(host.numpy()).cast('B')).hexdigest())
        for name,value in sorted(llm.engine.model.state_dict().items()):
            fingerprint('model.' + name, value)
        for bank,layers in llm.engine.moe_offload_cache.bank_sources.items():
            for layer,value in enumerate(layers):
                fingerprint(f'experts.{bank}.{layer}', value)
        result['loaded_tensor_hashes'] = hashes
        result['hash_scope'] = 'After startup timer; all model state_dict tensors and all source expert bank bytes'
    (a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
finally:
    llm.engine.shutdown()
