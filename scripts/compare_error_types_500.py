from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def load_dataset(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset file: {path}")


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def get_qid(row: dict, idx: int):
    return str(
        row.get("_id")
        or row.get("id")
        or row.get("question_id")
        or row.get("qid")
        or f"row_{idx}"
    )


def find_string_by_keys(obj: Any, keys: list[str]) -> str:
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        for v in obj.values():
            r = find_string_by_keys(v, keys)
            if r:
                return r
    elif isinstance(obj, list):
        for x in obj:
            r = find_string_by_keys(x, keys)
            if r:
                return r
    return ""


def get_question_from_dataset(row: dict):
    return row.get("question") or row.get("query") or ""


def get_gold_from_dataset(row: dict):
    for k in ["answer", "gold", "gold_answer", "final_answer"]:
        v = row.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, list) and v:
            return str(v[0])
    # MuSiQue sometimes uses answers list
    v = row.get("answers")
    if isinstance(v, list) and v:
        return str(v[0])
    return ""


def get_pred_from_predrow(row: dict):
    # common direct fields
    for k in [
        "prediction", "pred", "answer", "final_answer",
        "generated_answer", "output", "response"
    ]:
        v = row.get(k)
        if isinstance(v, str):
            return v
    # recursive fallback
    return find_string_by_keys(
        row,
        [
            "prediction", "pred", "final_answer",
            "generated_answer", "output", "response"
        ],
    )


def classify_error(question: str, gold: str, pred: str) -> str:
    q = question.lower()
    g = str(gold).lower()
    p = str(pred).strip().lower()

    if not p:
        return "empty_or_no_answer"

    if g in {"yes", "no"} or q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ")):
        return "boolean_or_yesno"

    if any(x in q for x in ["which", "earlier", "later", "first", "last", "older", "younger", "larger", "smaller", "higher", "lower", "same", "both"]):
        return "comparison_or_operator"

    if any(x in q for x in ["when", "year", "date", "born", "died", "founded", "released"]):
        return "temporal_or_numeric"

    if any(x in q for x in ["where", "country", "city", "located", "place", "born in", "from"]):
        return "location_or_admin"

    if any(x in q for x in ["father", "mother", "son", "daughter", "wife", "husband", "grandfather", "grandmother", "parent", "sibling"]):
        return "kinship_or_person_relation"

    if any(x in q for x in ["director", "writer", "author", "producer", "creator", "composer", "award", "occupation", "capital", "language"]):
        return "entity_attribute_relation"

    if any(x in q for x in ["film", "movie", "album", "song", "book", "novel", "series", "episode"]):
        return "work_title_relation"

    # answer granularity proxy: partial overlap but not exact
    if f1_score(pred, gold) > 0 and not exact_match(pred, gold):
        return "partial_or_granularity_mismatch"

    return "other"


def align_predictions(pred_rows: list[dict], dataset_rows: list[dict], limit: int):
    # map by id if possible
    ds_by_id = {}
    for i, r in enumerate(dataset_rows[:limit], 1):
        ds_by_id[get_qid(r, i)] = r

    aligned = []
    for i in range(limit):
        ds_row = dataset_rows[i] if i < len(dataset_rows) else {}
        pred_row = pred_rows[i] if i < len(pred_rows) else {}

        qid_pred = get_qid(pred_row, i + 1)
        ds_match = ds_by_id.get(qid_pred, ds_row)

        question = (
            find_string_by_keys(pred_row, ["question", "query"])
            or get_question_from_dataset(ds_match)
        )
        gold = (
            find_string_by_keys(pred_row, ["gold", "gold_answer", "reference"])
            or get_gold_from_dataset(ds_match)
        )
        pred = get_pred_from_predrow(pred_row)

        aligned.append({
            "qid": get_qid(ds_match, i + 1),
            "question": question,
            "gold": gold,
            "pred": pred,
            "raw_pred": pred_row,
        })

    return aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--pred", action="append", required=True,
                    help="Format: MethodName=/path/to/pred.jsonl")
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    dataset_rows = load_dataset(Path(args.dataset_path))[:args.limit]

    method_results = {}
    examples = defaultdict(list)

    for item in args.pred:
        if "=" not in item:
            raise ValueError(f"--pred must be Method=Path, got: {item}")
        method, path = item.split("=", 1)
        method = method.strip()
        pred_path = Path(path.strip())

        if not pred_path.exists():
            method_results[method] = {
                "missing": True,
                "path": str(pred_path),
            }
            continue

        pred_rows = load_jsonl(pred_path)
        aligned = align_predictions(pred_rows, dataset_rows, args.limit)

        total = len(aligned)
        correct = 0
        f1s = []
        buckets = defaultdict(int)
        bucket_total_questions = defaultdict(int)

        for ex in aligned:
            em = exact_match(ex["pred"], ex["gold"])
            f1 = f1_score(ex["pred"], ex["gold"])
            f1s.append(f1)
            if em:
                correct += 1
            else:
                cat = classify_error(ex["question"], ex["gold"], ex["pred"])
                buckets[cat] += 1
                if len(examples[(method, cat)]) < 5:
                    examples[(method, cat)].append({
                        "qid": ex["qid"],
                        "question": ex["question"],
                        "gold": ex["gold"],
                        "pred": ex["pred"],
                    })

        method_results[method] = {
            "missing": False,
            "path": str(pred_path),
            "total": total,
            "correct_em": correct,
            "wrong": total - correct,
            "EM": round(100 * correct / total, 2) if total else 0,
            "F1": round(100 * sum(f1s) / total, 2) if total else 0,
            "error_buckets": dict(sorted(buckets.items(), key=lambda x: (-x[1], x[0]))),
        }

    # collect all categories
    cats = sorted({
        c
        for r in method_results.values()
        if not r.get("missing")
        for c in r.get("error_buckets", {}).keys()
    })

    lines = []
    lines.append(f"# Error Type Analysis: {args.dataset_name}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Method | Total | EM | F1 | Wrong | Prediction File |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for method, r in method_results.items():
        if r.get("missing"):
            lines.append(f"| {method} | - | - | - | - | missing: `{r['path']}` |")
        else:
            lines.append(
                f"| {method} | {r['total']} | {r['EM']} | {r['F1']} | "
                f"{r['wrong']} | `{r['path']}` |"
            )

    lines.append("")
    lines.append("## Error Counts by Type")
    lines.append("")
    lines.append("| Error Type | " + " | ".join(method_results.keys()) + " |")
    lines.append("|---" + "|---:" * len(method_results) + "|")

    for cat in cats:
        row = [cat]
        for method, r in method_results.items():
            if r.get("missing"):
                row.append("-")
            else:
                row.append(str(r.get("error_buckets", {}).get(cat, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Error Rates by Type among All Questions (%)")
    lines.append("")
    lines.append("| Error Type | " + " | ".join(method_results.keys()) + " |")
    lines.append("|---" + "|---:" * len(method_results) + "|")

    for cat in cats:
        row = [cat]
        for method, r in method_results.items():
            if r.get("missing") or not r.get("total"):
                row.append("-")
            else:
                val = 100 * r.get("error_buckets", {}).get(cat, 0) / r["total"]
                row.append(f"{val:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Example Wrong Cases")
    for (method, cat), exs in examples.items():
        lines.append("")
        lines.append(f"### {method} / {cat}")
        for ex in exs:
            lines.append(f"- qid={ex['qid']}")
            lines.append(f"  - q: {ex['question']}")
            lines.append(f"  - gold: {ex['gold']}")
            lines.append(f"  - pred: {ex['pred']}")

    out_md = Path(args.output_md)
    out_json = Path(args.output_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(method_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] wrote {out_md}")
    print(f"[done] wrote {out_json}")
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
