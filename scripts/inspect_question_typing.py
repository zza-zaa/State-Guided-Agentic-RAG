from __future__ import annotations

import json
from pathlib import Path
import typer

from question_typing import infer_question_type

app = typer.Typer()

def load_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for x in data:
                yield x
        else:
            yield data

@app.command()
def main(
    dataset_path: Path = typer.Option(..., exists=True),
    limit: int = typer.Option(30),
    output_path: Path = typer.Option(...),
):
    rows = list(load_rows(dataset_path))[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as w:
        for i, row in enumerate(rows, 1):
            q = row.get("question", "")
            spec = infer_question_type(q)
            w.write(f"## {i}\n")
            w.write(f"Q: {q}\n")
            w.write(spec.to_json() + "\n\n")

    print(f"[done] wrote typing inspection to {output_path}")

if __name__ == "__main__":
    app()
