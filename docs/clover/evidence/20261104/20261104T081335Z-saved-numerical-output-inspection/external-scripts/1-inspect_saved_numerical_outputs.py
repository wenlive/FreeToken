"""Decode existing numerical evidence only; no model inference or new measurements."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
root=Path('C:/workspace/runs/T-20261104-clover-improve')
native_path=root/'20261104T054326Z-b1-native-noise-none/data/results.json'
hf_path=root/'20261104T054911Z-b1-hf-q8-progress/data/results.json'
tokenizer_path=Path('C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff/tokenizer.json')
tokenizer=Tokenizer.from_file(str(tokenizer_path))
native,hf=[json.loads(path.read_bytes()) for path in (native_path,hf_path)]
result=dict(state='complete',scope='Read and decode saved 64-token-cap outputs only; no model inference, no new quality score.',
    sources=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
             for path in (native_path,hf_path,tokenizer_path)],cases=[])
for reference in hf['cases']:
    rows=[r for r in native['runs'] if not r['warmup'] and r['prompt']==reference['prompt']
          and r['width']==reference['width'] and r['mode']=='denoise']
    assert len(rows)==1
    row=rows[0]
    assert native['prompts'][row['prompt']]['token_ids']==reference['prompt_ids']
    left,right=row['token_ids'],reference['visible_token_ids']
    result['cases'].append(dict(prompt=row['prompt'],width=row['width'],native_text=row['text'],
        hf_text=tokenizer.decode(right,skip_special_tokens=True),native_ids_equal_hf=left==right,
        native_length=len(left),hf_length=len(right),
        native_decode_matches_saved_text=tokenizer.decode(left,skip_special_tokens=True)==row['text'],
        first_difference=next((i for i,(x,y) in enumerate(zip(left,right)) if x!=y),
            min(len(left),len(right)) if len(left)!=len(right) else None)))
(a.output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
for case in result['cases']:
    print(json.dumps(case,ensure_ascii=False))
