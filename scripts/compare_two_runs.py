from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer()

def normalize(s: str | None) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())

@app.command()
def main(
    old_predictions: Path = typer.Option(..., exists=True),
    new_predictions: Path = typer.Option(..., exists=True),
    limit: int = typer.Option(10),
):
    old_map = {}
    new_map = {}

    with old_predictions.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            old_map[x.get("qid")] = x

    with new_predictions.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            new_map[x.get("qid")] = x

    old_right_new_wrong = []
    old_wrong_new_right = []

    qids = sorted(set(old_map) & set(new_map))
    for qid in qids:
        o = old_map[qid]
        n = new_map[qid]
        gold = normalize(o.get("gold_answer"))
        op = normalize(o.get("pred_answer"))
        np = normalize(n.get("pred_answer"))

        old_right = (op == gold and gold != "")
        new_right = (np == gold and gold != "")

        if old_right and not new_right:
            old_right_new_wrong.append((qid, o, n))
        elif not old_right and new_right:
            old_wrong_new_right.append((qid, o, n))

    print("=" * 140)
    print("OLD RIGHT -> NEW WRONG:", len(old_right_new_wrong))
    print("=" * 140)
    for qid, o, n in old_right_new_wrong[:limit]:
        print("-" * 140)
        print("qid:", qid)
        print("question:", o.get("question"))
        print("gold:", o.get("gold_answer"))
        print("[old] pred:", o.get("pred_answer"), "steps:", o.get("steps"))
        print("[new] pred:", n.get("pred_answer"), "steps:", n.get("steps"))

    print("=" * 140)
    print("OLD WRONG -> NEW RIGHT:", len(old_wrong_new_right))
    print("=" * 140)
    for qid, o, n in old_wrong_new_right[:limit]:
        print("-" * 140)
        print("qid:", qid)
        print("question:", o.get("question"))
        print("gold:", o.get("gold_answer"))
        print("[old] pred:", o.get("pred_answer"), "steps:", o.get("steps"))
        print("[new] pred:", n.get("pred_answer"), "steps:", n.get("steps"))

if __name__ == "__main__":
    app()
