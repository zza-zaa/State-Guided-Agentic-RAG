from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/raid/peiyu/csa/outputs")

rows = [
    ("patch24_typed_soft", ROOT / "hotpot_dev100_patch24_typed_soft_metrics.json"),
    ("patch25_typed_secondhop_100", ROOT / "hotpot_dev100_patch25_typed_secondhop_metrics.json"),
    ("patch25_typed_secondhop_300", ROOT / "hotpot_dev300_patch25_typed_secondhop_metrics.json"),
    ("patch26_answer_grounder_300", ROOT / "hotpot_dev300_patch26_answer_grounder_metrics.json"),
    ("patch26b_conservative_grounder_300", ROOT / "hotpot_dev300_patch26b_conservative_grounder_metrics.json"),
    ("patch27_target_selector_300", ROOT / "hotpot_dev300_patch27_target_selector_metrics.json"),
    ("patch28_path_scorer_300", ROOT / "hotpot_dev300_patch28_path_scorer_metrics.json"),
]

out = []
out.append("| module | total | EM | F1 | avg_steps | error_count |")
out.append("|---|---:|---:|---:|---:|---:|")

for name, path in rows:
    if not path.exists():
        out.append(f"| {name} | - | - | - | - | - |")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    out.append(
        f"| {name} | {data.get('total','-')} | "
        f"{data.get('EM','-')} | {data.get('F1','-')} | "
        f"{data.get('avg_steps','-')} | {data.get('error_count','-')} |"
    )

report = ROOT / "hotpot_module_report.md"
report.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"[done] wrote {report}")
