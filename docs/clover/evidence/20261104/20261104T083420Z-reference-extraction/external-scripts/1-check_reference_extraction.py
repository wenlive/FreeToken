"""Verify Clover tests with python/tests extracted and benchmark aliases in the full tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
source=Path.cwd()
extract=a.output/'extracted'
extract.mkdir()
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],text=True).split('\0')
inventory=[]
for name in sorted(set(files)):
    relative=Path(name)
    if not name or (relative.parts[0] not in ('python','tests') and name!='pyproject.toml'):
        continue
    original=source/relative
    if not original.is_file():
        continue
    target=extract/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original,target)
    inventory.append(dict(file=relative.as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
assert not (extract/'benchmarks').exists() and not (extract/'docs').exists()
selected=[p.relative_to(extract).as_posix() for p in sorted((extract/'tests').rglob('test_clover*.py'))]
env=dict(os.environ,PYTHONPATH=str(extract/'python')+os.pathsep+str(extract),CUDA_VISIBLE_DEVICES='-1',PYTORCH_NVML_BASED_CUDA_CHECK='0')
for key in ('CLOVER_GGUF_CHECKPOINT','CLOVER_HF_SOURCE'):
    env.pop(key,None)
probe="import freetoken, tests.clover_reference.oracles as r; from pathlib import Path; root=Path.cwd(); assert Path(freetoken.__file__).is_relative_to(root); assert Path(r.__file__).is_relative_to(root)"
subprocess.run([sys.executable,'-c',probe],cwd=extract,env=env,check=True)
command=[sys.executable,'-m','pytest','-q','-ra','--junitxml',str(a.output/'extraction-junit.xml'),*selected]
child=subprocess.run(command,cwd=extract,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
(a.output/'extraction-pytest.log').write_text(child.stdout,encoding='utf-8')
print(child.stdout,flush=True)
assert child.returncode==0,child.returncode
aliases="""
import importlib
for old,new in [('oracles','oracles'),('clover_hf_reference','hf'),('clover_gguf_reference','gguf'),('clover_gguf_decode','gguf_decode')]:
    alias=importlib.import_module('benchmarks.clover.'+old)
    canonical=importlib.import_module('tests.clover_reference.'+new)
    assert alias is canonical,(old,new)
    sentinel=object()
    alias._alias_probe=sentinel
    assert canonical._alias_probe is sentinel
    del alias._alias_probe
print('Four benchmark aliases preserve module identity and monkeypatch visibility')
"""
root_env=dict(env,PYTHONPATH=str(source/'python')+os.pathsep+str(source))
subprocess.run([sys.executable,'-c',aliases],cwd=source,env=root_env,check=True)
command_tools=[sys.executable,'-m','pytest','-q','-ra','benchmarks/clover/tests','--junitxml',str(a.output/'tools-junit.xml')]
tools=subprocess.run(command_tools,cwd=source,env=root_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
(a.output/'tools-pytest.log').write_text(tools.stdout,encoding='utf-8')
print(tools.stdout,flush=True)
assert tools.returncode==0,tools.returncode
(a.output/'results.json').write_text(json.dumps(dict(state='complete',extraction_command=command,
    tool_command=command_tools,files=inventory,core_summary=child.stdout.splitlines()[-1],
    tool_summary=tools.stdout.splitlines()[-1],aliases_preserve_module_identity=True),indent=2),encoding='utf-8')
