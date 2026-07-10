from __future__ import annotations

import json
from pathlib import Path
import typer

app = typer.Typer()

def norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())

def get_pred(row: dict) -> str:
    for k in ["pred", "pred_answer", "prediction"]:
        if k in row:
            return str(row.get(k, "") or "")
    return ""

def get_gold(row: dict) -> str:
    for k in ["gold", "gold_answer", "answer"]:
        if k in row:
            return str(row.get(k, "") or "")
    return ""

def classify(row: dict) -> str:
    q = norm(row.get("question", ""))
    pred = norm(get_pred(row))
    gold = norm(get_gold(row))
    state = row.get("state", {}) or {}
    slots = state.get("slots", []) or []
    conflicts = state.get("conflicts", []) or []

    bridge_slots = [s for s in slots if s.get("target_role") == "bridge"]
    answer_slots = [s for s in slots if s.get("target_role") == "answer_target"]

    if pred and gold and (gold in pred or pred in gold) and pred != gold:
        return "projection_or_granularity"

    if any(s.get("status") == "missing" for s in bridge_slots):
        return "bridge_missing"

    if any(s.get("status") == "missing" for s in answer_slots):
        return "second_hop_missing"

    if any(x in q for x in [
        "founded by", "who followed", "child of", "father of", "mother of",
        "which country", "what administrative territorial entity", "borders what county",
        "county", "league", "record label"
    ]):
        return "relation_direction_or_answer_type"

    if any(x in q for x in ["earlier", "later", "younger", "older", "died first", "both "]):
        return "comparison_or_operator"

    if conflicts:
        return "entity_or_relation_disambiguation"

    return "other"

@app.command()
def main(predictions: Path = typer.Option(..., exists=True),
         output: Path = typer.Option(...)):
    rows = []
    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    wrong = [r for r in rows if norm(get_pred(r)) != norm(get_gold(r))]

    counts = {}
    examples = {}
    for r in wrong:
        c = classify(r)
        counts[c] = counts.get(c, 0) + 1
        examples.setdefault(c, []).append(r)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as w:
        w.write("# Error Audit v2\n\n")
        w.write(f"total_rows: {len(rows)}\n")
        w.write(f"total_wrong: {len(wrong)}\n\n")
        for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            w.write(f"## {k}: {v}\n")
            for ex in examples[k][:5]:
                w.write(f"- qid={ex.get('qid')}\n")
                w.write(f"  - q: {ex.get('question')}\n")
                w.write(f"  - gold: {get_gold(ex)}\n")
                w.write(f"  - pred: {get_pred(ex)}\n")
            w.write("\n")

    print(f"[done] wrote audit to {output}")

if __name__ == "__main__":
    app()
