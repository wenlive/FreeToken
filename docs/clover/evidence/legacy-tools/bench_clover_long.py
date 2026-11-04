"""Run the ordinary Clover benchmark with saved long-prompt user text."""
import json
from pathlib import Path
import sys

import bench_clover

index = sys.argv.index("--long-prompts-json")
path = Path(sys.argv[index + 1])
del sys.argv[index:index + 2]
rows = json.loads(path.read_text(encoding="utf-8"))["prompts"]
bench_clover.PROMPTS.update({name: row["user_text"] for name, row in rows.items()})
bench_clover.main()
