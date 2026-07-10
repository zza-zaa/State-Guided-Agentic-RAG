from __future__ import annotations

import json
from pathlib import Path

FILES = {
    "Hotpot-100": "outputs/hotpot_dev100_crosscheck_metrics.json",
    "2Wiki-100": "outputs/2wiki_dev100_patch3_metrics.json",
    "MuSiQue-100": "outputs/musique_dev100_patch3_metrics.json",
}

print("| Dataset | Total | EM | F1 | Avg Steps | Error Count |")
print("|---|---:|---:|---:|---:|---:|")

for name, fp in FILES.items():
    path = Path(fp)
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        x = json.load(f)
    print(
        f"| {name} | {x.get('total', 0)} | "
        f"{x.get('EM', 0):.2f} | {x.get('F1', 0):.2f} | "
        f"{x.get('avg_steps', 0):.3f} | {x.get('error_count', 0)} |"
    )
