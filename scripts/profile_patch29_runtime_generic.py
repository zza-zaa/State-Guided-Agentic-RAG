from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

from csa_rag.agent.pipeline import CSARAGPipeline


def load_rows(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected top-level list in JSON file: {path}")
    else:
        raise ValueError(f"Unsupported dataset format: {path}")


def get_qid(row, idx: int):
    return (
        row.get("_id")
        or row.get("id")
        or row.get("question_id")
        or f"row_{idx}"
    )


def get_question(row):
    q = row.get("question")
    if not q:
        raise ValueError(f"Missing 'question' field in row: {row}")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--details-out", required=True)
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    models_config = Path(args.models_config)
    index_dir = Path(args.index_dir)
    summary_out = Path(args.summary_out)
    details_out = Path(args.details_out)

    rows = load_rows(dataset_path)
    rows = rows[args.offset: args.offset + args.limit]

    t0 = time.perf_counter()
    pipe = CSARAGPipeline(models_config=models_config, index_dir=index_dir)
    t1 = time.perf_counter()

    init_sec = t1 - t0
    per_q = []
    details = []
    fail_count = 0

    for i, row in enumerate(rows, 1):
        qid = get_qid(row, i)
        q = get_question(row)

        qs = time.perf_counter()
        try:
            result = pipe.run(q)
            qe = time.perf_counter()
            dt = qe - qs
            per_q.append(dt)
            details.append({
                "idx": i,
                "qid": qid,
                "question": q,
                "elapsed_sec": dt,
                "steps": getattr(result, "steps", None),
                "final_answer": getattr(result, "final_answer", None),
                "status": "ok",
            })
        except Exception as e:
            qe = time.perf_counter()
            dt = qe - qs
            fail_count += 1
            details.append({
                "idx": i,
                "qid": qid,
                "question": q,
                "elapsed_sec": dt,
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    details_out.parent.mkdir(parents=True, exist_ok=True)

    with details_out.open("w", encoding="utf-8") as f:
        for x in details:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    success_count = len(per_q)
    summary = {
        "dataset_path": str(dataset_path),
        "models_config": str(models_config),
        "index_dir": str(index_dir),
        "limit": args.limit,
        "offset": args.offset,
        "init_sec": init_sec,
        "success_count": success_count,
        "fail_count": fail_count,
        "qa_loop_sec_success_only": sum(per_q),
        "avg_sec_per_question_excl_init_success_only": (sum(per_q) / success_count) if success_count else None,
        "p50_sec_per_question_excl_init_success_only": statistics.median(per_q) if success_count else None,
        "p90_sec_per_question_excl_init_success_only": sorted(per_q)[max(0, int(0.9 * success_count) - 1)] if success_count else None,
        "max_sec_per_question_excl_init_success_only": max(per_q) if success_count else None,
        "min_sec_per_question_excl_init_success_only": min(per_q) if success_count else None,
    }

    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
