"""Probe a near-8K single request through Speed's chunked-prefill settings."""
from pathlib import Path
from datetime import datetime, timezone
import json, time
import torch
from freetoken.llm import LLM
from freetoken.core import SamplingParams
from freetoken.mm.config import MultimodalConfig

TASK=Path(r'C:\workspace\tasks\T-20261028-lora-opt')
RUN=Path(r'C:\workspace\runs\T-20261028-lora-opt')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-long-prefill')
RUN.mkdir()
(TASK/'long-prefill-run.txt').write_text(str(RUN))
profile=json.loads((TASK/'speed-expert-r2-profile.json').read_text())
result={'status':'RUNNING','profile':profile,'purpose':'Capacity/chunk continuation only; no answer accuracy claim'}
try:
    with torch.inference_mode():
        llm=LLM(profile['model'], mm=MultimodalConfig(disabled_encoders=frozenset({'vision','audio'})), **profile['settings'])
        message='容量检查编号982。请概括下面的工程记录。\n'+('读取专家缓存后按原始专家编号计算低秩增量，保持路由权重和共享专家的计算顺序。\n'*400)
        tokens=llm.tokenizer.apply_chat_template([{'role':'user','content':message}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        # Retain the real chat-template generation suffix while setting an exact allocation stress length.
        tokens=tokens[:7160]+tokens[-8:]
        assert len(tokens)==7168
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        answer=llm.generate([tokens],SamplingParams(temperature=0,ignore_eos=True,max_tokens=128))[0]
        torch.cuda.synchronize()
        result.update(input_tokens=len(tokens),input_token_ids=tokens,answer=answer,
                      wall_seconds=time.perf_counter()-start,peak_reserved=torch.cuda.max_memory_reserved(),
                      free_total=list(torch.cuda.mem_get_info()))
        assert len(answer['token_ids'])==128
        result['status']='PASS'
finally:
    (RUN/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k in ['status','input_tokens','wall_seconds','peak_reserved','free_total']}))
