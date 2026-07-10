from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def main(predictions: Path = typer.Option(..., exists=True)):
    total = 0
    total_slots = 0
    total_confirmed = 0
    total_missing = 0
    total_conflicts = 0
    answer_target_confirmed = 0

    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state = row.get("state", {}) or {}
            slots = state.get("slots", []) or []
            conflicts = state.get("conflicts", []) or []
            missing = state.get("missing_slots", []) or []

            total += 1
            total_slots += len(slots)
            total_missing += len(missing)
            total_conflicts += len(conflicts)

            confirmed = 0
            has_answer_target_confirmed = False
            for s in slots:
                if s.get("status") == "confirmed":
                    confirmed += 1
                if s.get("target_role") == "answer_target" and s.get("status") == "confirmed":
                    has_answer_target_confirmed = True
            total_confirmed += confirmed
            if has_answer_target_confirmed:
                answer_target_confirmed += 1

    if total == 0:
        print("No rows found.")
        return

    print("total =", total)
    print("avg_slots =", round(total_slots / total, 4))
    print("avg_confirmed_slots =", round(total_confirmed / total, 4))
    print("avg_missing_slots =", round(total_missing / total, 4))
    print("avg_conflicts =", round(total_conflicts / total, 4))
    print("answer_target_confirmed_rate =", round(answer_target_confirmed / total, 4))

if __name__ == "__main__":
    app()
