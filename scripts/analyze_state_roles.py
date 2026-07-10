from __future__ import annotations

import json
from pathlib import Path
import typer

app = typer.Typer()

def depth(slot_id, slot_map, memo):
    if slot_id in memo:
        return memo[slot_id]
    slot = slot_map.get(slot_id)
    if slot is None or not slot.get("depends_on"):
        memo[slot_id] = 0
        return 0
    d = 1 + max(depth(dep, slot_map, memo) for dep in slot.get("depends_on", []))
    memo[slot_id] = d
    return d

@app.command()
def main(predictions: Path = typer.Option(..., exists=True)):
    total = 0
    no_answer_target = 0
    answer_target_confirmed = 0
    complex_q = 0
    complex_answer_target_confirmed = 0
    avg_slots = 0
    avg_missing = 0
    avg_conflicts = 0

    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            state = row.get("state", {}) or {}
            slots = state.get("slots", []) or []
            slot_map = {s.get("slot_id"): s for s in slots}
            memo = {}

            total += 1
            avg_slots += len(slots)
            avg_missing += len(state.get("missing_slots", []) or [])
            avg_conflicts += len(state.get("conflicts", []) or [])

            targets = [s for s in slots if s.get("target_role") == "answer_target"]
            if not targets:
                no_answer_target += 1
            else:
                if any(s.get("status") == "confirmed" for s in targets):
                    answer_target_confirmed += 1

            max_depth = 0
            for sid in slot_map:
                max_depth = max(max_depth, depth(sid, slot_map, memo))
            is_complex = (len(slots) >= 4) or (max_depth >= 2)
            if is_complex:
                complex_q += 1
                if targets and any(s.get("status") == "confirmed" for s in targets):
                    complex_answer_target_confirmed += 1

    if total == 0:
        print("No rows.")
        return

    print("total =", total)
    print("avg_slots =", round(avg_slots / total, 4))
    print("avg_missing_slots =", round(avg_missing / total, 4))
    print("avg_conflicts =", round(avg_conflicts / total, 4))
    print("no_answer_target =", no_answer_target)
    print("answer_target_confirmed_rate =", round(answer_target_confirmed / total, 4))
    print("complex_questions =", complex_q)
    if complex_q > 0:
        print("complex_answer_target_confirmed_rate =", round(complex_answer_target_confirmed / complex_q, 4))
