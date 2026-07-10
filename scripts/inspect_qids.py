from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def main(
    predictions: Path = typer.Option(..., exists=True),
    qids: str = typer.Option(..., help="Comma-separated qids"),
):
    want = {x.strip() for x in qids.split(",") if x.strip()}
    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x.get("qid") in want:
                print("=" * 140)
                print("qid:", x.get("qid"))
                print("question:", x.get("question"))
                print("gold:", x.get("gold_answer"))
                print("pred:", x.get("pred_answer"))
                print("raw_pred:", x.get("raw_pred_answer"))
                print("steps:", x.get("steps"))
                print("state:", json.dumps(x.get("state", {}), ensure_ascii=False, indent=2))
                print("retrieved_evidence_top5:")
                for e in x.get("retrieved_evidence", [])[:5]:
                    print(json.dumps(e, ensure_ascii=False))
                print("rationale:", x.get("rationale"))

if __name__ == "__main__":
    app()
