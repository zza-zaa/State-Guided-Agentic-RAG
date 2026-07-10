from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer()

def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())

@app.command()
def main(
    predictions: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
):
    rows = []
    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            pred = normalize(ex.get("pred_answer", ""))
            gold = normalize(ex.get("gold_answer", ""))
            if pred != gold:
                rows.append(ex)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"errors={len(rows)} written to {output}")

if __name__ == "__main__":
    app()
