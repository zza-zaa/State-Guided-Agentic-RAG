from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import typer

app = typer.Typer()

def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

@app.command()
def main(
    errors: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
):
    rows = []
    counter = Counter()

    with errors.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            pred = normalize_text(ex.get("pred_answer", ""))
            gold = normalize_text(ex.get("gold_answer", ""))
            steps = ex.get("steps", -1)
            state = ex.get("state", {})
            missing = state.get("missing_slots", [])
            conflicts = state.get("conflicts", [])

            if pred == "" or pred in {"unknown", "it cannot be determined"}:
                error_type = "answer_failure"
            elif missing:
                error_type = "second_hop_missing"
            elif conflicts:
                error_type = "conflict_unresolved"
            elif steps >= 4:
                error_type = "multi_step_failure"
            else:
                error_type = "answer_not_canonical"

            ex["error_type"] = error_type
            rows.append(ex)
            counter[error_type] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("==== ERROR TYPE COUNTS ====")
    for k, v in counter.items():
        print(f"{k}\t{v}")

if __name__ == "__main__":
    app()
