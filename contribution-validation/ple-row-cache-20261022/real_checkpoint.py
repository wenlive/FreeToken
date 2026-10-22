import json
from pathlib import Path
import random

import torch
from freetoken.kernel import _row_store
from freetoken.models.qwen4_exp.ple_disk import source_from_gguf
from freetoken.server.args import parse_args

run = Path(__file__).parent
output = run / 'real-checkpoint01.json'
if output.exists():
    raise SystemExit('Output exists')
model = next(Path('C:/workspace/models/siltok-speed/flash-next/sources/ISTA-DASLab').rglob('*00001-of-00002.gguf'))
config, _ = parse_args(['--model', str(model), '--ple-row-cache-rows', '65536', '--attn', 'triton'])
assert config.ple_row_cache_rows == 65536
source = source_from_gguf(str(model))
store = _row_store.RowStore(source.paths, source.extent_file, source.extent_base,
                            source.rows_per_extent, source.row_bytes, source.row_stride, False)
generator = random.Random(29)
ids = [0, 1, source.total_rows - 1] + [generator.randrange(source.total_rows) for _ in range(61)]
index = torch.tensor(ids, dtype=torch.int64)
expected = []
for row in ids:
    extent, relative = divmod(row, source.rows_per_extent)
    with open(source.paths[source.extent_file[extent]], 'rb') as file:
        file.seek(source.extent_base[extent] + relative * source.row_stride)
        expected.append(file.read(source.row_bytes))
want = b''.join(expected)
stats = []
for capacity in (0, 65536, 8, 0):
    store.set_row_cache(capacity)
    for _ in range(3):
        staging = torch.empty(len(ids) * source.row_bytes, dtype=torch.uint8)
        store.stage_rows(index.data_ptr(), len(ids), staging.data_ptr())
        store.flush()
        assert staging.numpy().tobytes() == want
    stats.append(store.io_stats())
report = dict(model=str(model), model_id='ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
              revision='ed59f92082b1e93c0e96d60a8b11aab089b52f09',
              table=dict(rows=source.total_rows, row_bytes=source.row_bytes,
                         ggml_type=source.ggml_type, row_values=source.row_values,
                         offsets=source.extent_base),
              checked_rows=len(ids), reads=12, bytes_exact=True, cli_capacity=config.ple_row_cache_rows,
              stats=stats, scope='real checkpoint packed row reads, not full-model inference')
output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
