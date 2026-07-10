from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import typer

app = typer.Typer()


def normalize_answer(s: str) -> str:
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@app.command()
def main(
    predictions: Path = typer.Option(..., exists=True, help="Prediction jsonl file"),
    output: Path = typer.Option(..., help="Metrics json output"),
):
    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    error_count = 0
    total_steps = 0
    valid_steps = 0

    examples: list[dict[str, Any]] = []

    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(ex)

    for ex in examples:
        total += 1
        gold = ex.get("gold_answer", "")
        pred = ex.get("pred_answer", "")

        if ex.get("error") is not None:
            error_count += 1

        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)

        em_sum += em
        f1_sum += f1

        steps = ex.get("steps", -1)
        if isinstance(steps, int) and steps >= 0:
            total_steps += steps
            valid_steps += 1

    metrics = {
        "total": total,
        "error_count": error_count,
        "EM": round(100.0 * em_sum / total, 2) if total else 0.0,
        "F1": round(100.0 * f1_sum / total, 2) if total else 0.0,
        "avg_steps": round(total_steps / valid_steps, 3) if valid_steps else None,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
