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

def family(q: str) -> str:
    q = norm(q)
    if any(x in q for x in ["father-in-law", "mother-in-law", "paternal", "maternal", "child of", "father of", "mother of", "spouse"]):
        return "kinship"
    if any(x in q for x in ["earlier", "later", "younger", "older", "died first", "both "]):
        return "comparison_or_boolean"
    if any(x in q for x in ["county", "administrative territorial entity", "borders what county", "where is", "where was", "place of birth"]):
        return "location_admin"
    if any(x in q for x in ["award", "employer", "headquartered", "record label", "league"]):
        return "entity_attribute"
    return "other"

@app.command()
def main(predictions: Path = typer.Option(..., exists=True),
         output: Path = typer.Option(...)):
    rows = []
    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    stats = {}
    for r in rows:
        fam = family(r.get("question", ""))
        stats.setdefault(fam, {"total": 0, "correct": 0})
        stats[fam]["total"] += 1
        if norm(get_pred(r)) == norm(get_gold(r)):
            stats[fam]["correct"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as w:
        w.write("| family | total | correct | acc |\n")
        w.write("|---|---:|---:|---:|\n")
        for fam, s in sorted(stats.items()):
            acc = 100.0 * s["correct"] / max(1, s["total"])
            w.write(f"| {fam} | {s['total']} | {s['correct']} | {acc:.2f} |\n")

    print(f"[done] wrote breakdown to {output}")

if __name__ == "__main__":
    app()
