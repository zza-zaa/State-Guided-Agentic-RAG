from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def main(
    input1: Path = typer.Option(..., exists=True),
    input2: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
):
    rows = {}
    for path in [input1, input2]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                rows[ex["qid"]] = ex

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for qid in sorted(rows):
            f.write(json.dumps(rows[qid], ensure_ascii=False) + "\n")

    print(f"merged {len(rows)} rows into {output}")

if __name__ == "__main__":
    app()
