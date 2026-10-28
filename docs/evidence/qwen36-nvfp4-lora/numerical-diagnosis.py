from pathlib import Path
from datetime import datetime,timezone
import ast,importlib.util,json,os,hashlib
import torch

ROOT=Path(r'C:\workspace\wt\T-20261028-freetoken-lora\FreeToken')
fixture_source=Path(os.environ.get('NUMERICAL_FIXTURE_SOURCE',str(ROOT/'tests/lora/test_nvfp4.py')))
spec=importlib.util.spec_from_file_location('fixture',fixture_source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tree=ast.parse(fixture_source.read_text())
fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name.startswith('test_'))
body=[]
for node in fn.body:
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.testing.assert_close':break
    body.append(node)
scope=dict(module.__dict__,tokens=17,h=2048,i=512)
exec(compile(ast.Module(body=body,type_ignores=[]),'fixed-fixture','exec'),scope)
globals().update({k:v for k,v in scope.items() if not k.startswith('__')})
from freetoken.lora.ops import gate_up_residual
from freetoken.moe.fused_nvfp4 import _prefill_config,_prefill_gemm
from freetoken.moe.fused import moe_align_block_size
cfg=_prefill_config(tokens)
sorted_ids,expert_ids,ntpp=moe_align_block_size(gi,cfg['BLOCK_SIZE_M'],e)
delta=gate_up_residual(gx,gi,gpu_pairs)
first=torch.empty(tokens,k,2*i,device='cuda',dtype=torch.float32)
_prefill_gemm(gx,*banks[:3],first,weights.reshape(-1),sorted_ids,expert_ids,ntpp,tokens*k,k,False,cfg,delta)
cpu_first=[]
for t in range(tokens):
    routes=[]
    for j in range(k):
        ex=int(ids[t,j]);parts=[]
        for index,name in enumerate(['gate','up']):
            p=pairs[name]
            parts.append(wg[ex,index*i:(index+1)*i]@x[t].float()+p.b[ex]@(p.a[ex]@x[t].float())*p.scaling)
        routes.append(torch.cat(parts))
    cpu_first.append(torch.stack(routes))
cpu_first=torch.stack(cpu_first)
gpu_first=first.cpu()
rounded_cpu=cpu_first.bfloat16();rounded_gpu=gpu_first.bfloat16()
def metrics(a,b):
    d=(a.float()-b.float());return dict(relative_l2=(d.norm()/b.float().norm()).item(),max_abs=d.abs().max().item(),max_reference=b.abs().max().item(),different_elements=int((d!=0).sum()),total=d.numel())
final=metrics(prefill.cpu(),expected)
first_metrics=metrics(gpu_first,cpu_first)
round_metrics=metrics(rounded_gpu,rounded_cpu)
# Recompute the independent dense reference with only the first accumulator's
# BF16 boundary conditioned on the measured accumulator (not its downstream ops).
conditioned=torch.zeros(tokens,h)
for t in range(tokens):
    routes=[]
    for j in range(k):
        ex=int(ids[t,j]);g=rounded_gpu[t,j,:i].float();u=rounded_gpu[t,j,i:].float()
        act=(torch.nn.functional.silu(g)*u).bfloat16().float();p=pairs['down']
        out=wd[ex]@act+p.b[ex]@(p.a[ex]@act)*p.scaling
        routes.append((out*weights[t,j].cpu()).bfloat16().float())
    conditioned[t]=torch.stack(routes).sum(0)
result={'fixture':{'tokens':17,'hidden':2048,'intermediate':512,'experts':5,'top_k':2,'seed':81},
        'fixture_source_sha256':hashlib.sha256(fixture_source.read_bytes()).hexdigest(),
        'unconditioned_final':final,'first_accumulator_fp32':first_metrics,'first_accumulator_bf16':round_metrics,
        'first_boundary_conditioned_final':metrics(prefill.cpu(),conditioned.bfloat16()),
        'note':'Diagnostic preserves the original independent CPU reference and its pointwise failure.'}
out=Path(r'C:\workspace\runs\T-20261028-freetoken-lora')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-numerical-diagnosis')
out.mkdir();(out/'results.json').write_text(json.dumps(result,indent=2))
(out/'fixture-source.py').write_bytes(fixture_source.read_bytes())
(out/'diagnose-source.py').write_bytes(Path(__file__).read_bytes())
Path(r'C:\workspace\tasks\T-20261028-freetoken-lora\numerical-diagnosis-run.txt').write_text(str(out))
print(json.dumps(result))
