from __future__ import annotations

import json
import time
import statistics
import traceback
from pathlib import Path

from csa_rag.agent.pipeline import CSARAGPipeline

DATASET_PATH = Path("/mnt/raid/peiyu/dataset/hotpot/hotpot_dev_distractor_v1.json")
MODELS_CONFIG = Path("configs/models.yaml")
INDEX_DIR = Path("data/indexes/hotpot")
OUTPUT_JSON = Path("outputs/hotpot_patch29_runtime_profile_robust.json")
OUTPUT_JSONL = Path("outputs/hotpot_patch29_runtime_profile_robust_details.jsonl")

LIMIT = 100
OFFSET = 0

def main():
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = rows[OFFSET: OFFSET + LIMIT]

    t0 = time.perf_counter()
    pipe = CSARAGPipeline(models_config=MODELS_CONFIG, index_dir=INDEX_DIR)
    t1 = time.perf_counter()

    init_sec = t1 - t0
    per_q = []
    details = []
    fail_count = 0

    for i, row in enumerate(rows, 1):
        qid = row.get("_id", row.get("id", str(i)))
        q = row["question"]

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

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for x in details:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    success_count = len(per_q)
    summary = {
        "limit": LIMIT,
        "offset": OFFSET,
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

    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
