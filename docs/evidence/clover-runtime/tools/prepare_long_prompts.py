"""Save exact 1024/4096-token no-thinking prompts for both frameworks."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

source = "C:/workspace/models/inputs/clover-35b-a3b-bf16/4b0402b65e523d0183a14c431955438714cb16ff"
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
output = Path(__file__).parent / "long-prompts"
output.mkdir(exist_ok=False)
rows = {}
for name, target in (("Long1K", 1024), ("Long4K", 4096)):
    full = "Summarize the following material in two sentences.\n" + "Trees cool city streets through shade and evaporation. " * 1200
    body = tokenizer.encode(full, add_special_tokens=False)
    for count in range(target - 100, target):
        user = tokenizer.decode(body[:count], skip_special_tokens=False)
        text = tokenizer.apply_chat_template([dict(role="user", content=user)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == target:
            break
    else:
        raise ValueError("Cannot construct exact prompt length")
    path = output / (name + ".txt")
    path.write_bytes(text.encode("utf-8"))
    rows[name] = dict(text=text, user_text=user, prompt_ids=ids, tokens=len(ids),
                     file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(output / "manifest.json").write_text(json.dumps(dict(state="complete", prompts=rows), indent=2), encoding="utf-8")
print({name: row["tokens"] for name, row in rows.items()})
