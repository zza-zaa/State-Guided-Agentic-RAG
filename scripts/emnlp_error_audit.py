from __future__ import annotations

import json
import re
from pathlib import Path
import typer

app = typer.Typer()

def norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())

def classify(row: dict) -> str:
    q = norm(row.get("question", ""))
    pred = norm(row.get("pred", ""))
    gold = norm(row.get("gold", ""))
    state = row.get("state", {}) or {}
    slots = state.get("slots", []) or []

    missing = state.get("missing_slots", []) or []
    conflicts = state.get("conflicts", []) or []

    # 1. projection / alias / granularity
    if pred and gold:
        if gold in pred or pred in gold:
            return "projection_or_granularity"
        if any(x in q for x in ["where was", "place of birth", "county", "when", "what year"]):
            return "projection_or_granularity"

    # 2. bridge missing
    bridge_slots = [s for s in slots if s.get("target_role") == "bridge"]
    if any(s.get("status") == "missing" for s in bridge_slots):
        return "bridge_missing"

    # 3. second-hop missing
    target_slots = [s for s in slots if s.get("target_role") == "answer_target"]
    if target_slots and any(s.get("status") == "missing" for s in target_slots):
        return "second_hop_missing"

    # 4. relation direction / answer type
    if any(x in q for x in [
        "founded by", "who followed", "which country", "what administrative territorial entity",
        "borders what county", "child of", "father of", "mother of", "paternal grandfather"
    ]):
        return "relation_direction_or_answer_type"

    # 5. comparison / operator misuse
    if any(x in q for x in ["earlier", "later", "younger", "older", "died first", "both "]):
        return "comparison_or_operator"

    # 6. disambiguation
    if conflicts:
        return "entity_or_relation_disambiguation"

    return "other"

@app.command()
def main(
    predictions: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
):
    rows = []
    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    wrong = [r for r in rows if norm(r.get("pred")) != norm(r.get("gold"))]

    counts = {}
    examples = {}
    for r in wrong:
        c = classify(r)
        counts[c] = counts.get(c, 0) + 1
        examples.setdefault(c, []).append({
            "qid": r.get("qid"),
            "question": r.get("question"),
            "gold": r.get("gold"),
            "pred": r.get("pred"),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as w:
        w.write("# Error Audit\n\n")
        total = len(wrong)
        w.write(f"total_wrong: {total}\n\n")
        for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            w.write(f"## {k}: {v}\n")
            for ex in examples[k][:5]:
                w.write(f"- qid={ex['qid']}\n")
                w.write(f"  - q: {ex['question']}\n")
                w.write(f"  - gold: {ex['gold']}\n")
                w.write(f"  - pred: {ex['pred']}\n")
            w.write("\n")

    print(f"[done] wrote audit to {output}")

if __name__ == "__main__":
    app()
