from __future__ import annotations

import json
from pathlib import Path
import typer

from question_typing import infer_question_type
from question_typing_v2 import infer_question_type_v2
from question_typing_v3 import infer_question_type_v3

app = typer.Typer()

VERSIONS = {
    "v1": infer_question_type,
    "v2": infer_question_type_v2,
    "v3": infer_question_type_v3,
}

FIELDS = [
    "relation_family",
    "operator_type",
    "answer_type",
    "answer_granularity",
    "chain_depth_hint",
]

@app.command()
def main(
    config_path: Path = typer.Option(..., exists=True),
    output_path: Path = typer.Option(...),
):
    cases = json.loads(config_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for version, fn in VERSIONS.items():
        total = len(cases)
        field_hits = {f: 0 for f in FIELDS}
        exact = 0

        lines.append(f"# {version}\n")
        for i, case in enumerate(cases, 1):
            q = case["question"]
            expect = case["expect"]
            pred = fn(q)

            matched_all = True
            for f, gold in expect.items():
                pred_val = getattr(pred, f)
                if pred_val == gold:
                    if f in field_hits:
                        field_hits[f] += 1
                else:
                    matched_all = False

            if matched_all:
                exact += 1
            else:
                lines.append(f"## fail {i}\n")
                lines.append(f"Q: {q}\n")
                lines.append(f"expect: {json.dumps(expect, ensure_ascii=False)}\n")
                lines.append("pred: " + pred.to_json() + "\n")

        lines.insert(len(lines), "")
        lines.append(f"exact_match: {exact}/{total} = {100.0*exact/total:.2f}%\n")
        for f, hit in field_hits.items():
            # only count fields that appear in at least one expectation
            denom = sum(1 for c in cases if f in c["expect"])
            if denom > 0:
                lines.append(f"{f}: {hit}/{denom} = {100.0*hit/denom:.2f}%\n")
        lines.append("\n---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote regression report to {output_path}")

if __name__ == "__main__":
    app()
