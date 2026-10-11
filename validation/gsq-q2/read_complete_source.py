"""Read all real routed-expert bytes without allocating the full bank or running inference."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from freetoken.models.qwen4_exp.q2_source import iter_q2_expert_pieces

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
config = json.loads((args.model/'config.json').read_text(encoding='utf-8'))
text = config.get('text_config', config)
source = config['freetoken_expert_source']
model = SimpleNamespace(expert_source=source, num_layers=text['num_hidden_layers'],
                        num_experts=text['num_experts'], hidden_size=text['hidden_size'],
                        moe_intermediate_size=text['moe_intermediate_size'])
started = time.perf_counter()
total, covered = 0, {}
for layer, first, last, pieces in iter_q2_expert_pieces(args.model, model):
    assert first == covered.get(layer, 0)
    covered[layer] = last
    total += sum(piece.numel() for piece in pieces.values())
expected = 3*model.num_layers*model.num_experts*model.hidden_size*model.moe_intermediate_size//64*18
assert covered == {layer:model.num_experts for layer in range(model.num_layers)}
assert total == expected
result = {'scope':'complete real expert source read only; no inference or full bank allocation',
          'layers':model.num_layers,'experts_per_layer':model.num_experts,'bytes':total,
          'seconds':time.perf_counter()-started,'source_sha256_receipt':source['sha256'],
          'source_size':source['size'],'passed':True}
args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
