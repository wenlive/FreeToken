"""Use Speed's real backend, IPC worker, release and AOT checks; no HTTP layer."""
from pathlib import Path
from datetime import datetime, timezone
import json, time
from silitok_speed.backends.base import BackendConfig
from silitok_speed.backends.freetoken import FreeTokenBackend
from silitok_speed.types import ModelSpec

TASK = Path(r'C:\workspace\tasks\T-20261028-lora-opt')
RUN = Path(r'C:\workspace\runs\T-20261028-lora-opt')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-speed-worker')
RUN.mkdir()
(TASK/'speed-worker-run.txt').write_text(str(RUN))
model = r'C:\workspace\models\inputs\qwen3.6-35b-a3b-nvfp4\1355db6a052410cfd62085d94b58866fd0f2c3c5'
adapter = r'C:\workspace\artifacts\nvfp4lora\0.0.1\20261028T024826Z\expert-r2\publish'
cache = Path(r'C:\workspace\tasks\T-20261028-freetoken-lora\runtime-kernel-cache-dir.txt').read_text().strip()
cfg = BackendConfig(extra={'kernel_cache_dir': cache, 'lora_path': adapter})
spec = ModelSpec(model_id='qwen3.6-35b-a3b', weights_path=model, quantization='nvfp4', fine_tune_tag='expert-r2-static')
backend = FreeTokenBackend()
result = {'status': 'RUNNING', 'adapter': adapter, 'requests': []}
try:
    start = time.perf_counter()
    backend.load(spec, cfg)
    result.update(load_seconds=time.perf_counter()-start, worker_pid=backend.worker_pid,
                  cache_geometry=backend.cache_geometry(), expert_storage=backend.expert_storage())
    corpus = Path(r'C:\workspace\data\datasets\nvfp4lora\canary-replay-v1')
    for filename, index in [('canary-holdout.jsonl', 11), ('replay.jsonl', 11)]:
        item = json.loads((corpus/filename).read_text(encoding='utf-8').splitlines()[index])
        # Dataset stores messages including the assistant reference.
        messages = [m for m in item['messages'] if m['role'] != 'assistant']
        expected = next(m['content'] for m in item['messages'] if m['role']=='assistant')
        start = time.perf_counter()
        events = list(backend.generate({'messages': messages, 'enable_thinking': False}, temperature=0.0, max_tokens=32))
        final = next(e for e in events if e['type']=='final')
        record = {'dataset': filename, 'index': index, 'expected': expected, 'messages': messages,
                  'wall_seconds': time.perf_counter()-start, 'events': events}
        result['requests'].append(record)
        assert final['text'].strip() == expected.strip(), record
    result['status'] = 'PASS'
finally:
    backend.unload()
    result['unloaded_state'] = backend.state.value
    result['worker_pid_after_unload'] = backend.worker_pid
    (RUN/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': result['status'], 'run': str(RUN), 'load_seconds': result['load_seconds']}))
