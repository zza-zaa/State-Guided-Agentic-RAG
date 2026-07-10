from __future__ import annotations

import re
from pathlib import Path
import typer

from question_typing import infer_question_type
from question_typing_v2 import infer_question_type_v2

app = typer.Typer()

def extract_questions(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    qs = re.findall(r"- q:\s*(.+)", text)
    return qs

@app.command()
def main(
    input_md: Path = typer.Option(..., exists=True),
    output_md: Path = typer.Option(...),
):
    qs = extract_questions(input_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_md.open("w", encoding="utf-8") as w:
        for i, q in enumerate(qs, 1):
            s1 = infer_question_type(q)
            s2 = infer_question_type_v2(q)
            w.write(f"## {i}\n")
            w.write(f"Q: {q}\n\n")
            w.write("### v1\n")
            w.write(s1.to_json() + "\n\n")
            w.write("### v2\n")
            w.write(s2.to_json() + "\n\n")

    print(f"[done] wrote comparison to {output_md}")

if __name__ == "__main__":
    app()
