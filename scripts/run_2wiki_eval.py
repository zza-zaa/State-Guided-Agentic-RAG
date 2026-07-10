from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csa_rag.agent.pipeline import CSARAGPipeline
from csa_rag.utils.answer_utils import canonicalize_answer
from csa_rag.utils.direct_answer import direct_answer_from_state
from csa_rag.utils.operator_gate import should_use_operator_answer
from dataset_utils import load_json_or_jsonl, normalize_2wiki_record

app = typer.Typer()

@app.command()
def main(
    dataset_path: Path = typer.Option(..., exists=True),
    index_dir: Path = typer.Option(..., exists=True),
    models_config: Path = typer.Option(..., exists=True),
    output_path: Path = typer.Option(...),
    limit: int = typer.Option(100),
    offset: int = typer.Option(0),
):
    pipe = CSARAGPipeline(models_config=models_config, index_dir=index_dir)

    rows = list(load_json_or_jsonl(dataset_path))
    rows = rows[offset: offset + limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".errors.log")

    success = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out, log_path.open("w", encoding="utf-8") as errlog:
        for i, rec in enumerate(rows, 1):
            x = normalize_2wiki_record(rec)

            try:
                question = str(x.get("question", "")).strip()
                gold = str(x.get("answer", "")).strip()
                qid = x.get("qid", "")

                if not question:
                    raise ValueError("empty question")

                last_error = None
                result = None
                for attempt in range(2):
                    try:
                        result = pipe.run(question)
                        break
                    except Exception as e:
                        last_error = e
                        msg = repr(e)
                        if attempt == 0 and ("JSONDecodeError" in msg or "Expecting" in msg):
                            time.sleep(1.0)
                            continue
                        break

                if result is None:
                    raise last_error if last_error is not None else ValueError("unknown failure")

                state_dict = result.state.model_dump()
                direct = direct_answer_from_state(question, state_dict)
                if should_use_operator_answer(question, state_dict, direct, result.final_answer):
                    base_pred = direct
                else:
                    base_pred = result.final_answer

                pred = canonicalize_answer(question, base_pred, state_dict)

                row = {
                    "qid": qid,
                    "question": question,
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "raw_pred_answer": result.final_answer,
                    "steps": result.steps,
                    "state": state_dict,
                    "retrieved_evidence": [e.model_dump() for e in result.retrieved_evidence],
                    "rationale": result.rationale,
                    "error": None,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                success += 1

            except Exception as e:
                failed += 1
                errlog.write("=" * 120 + "\n")
                errlog.write(f"[row {i}] qid={x.get('qid','')}\n")
                errlog.write(f"question={x.get('question','')}\n")
                errlog.write(f"error={repr(e)}\n")
                errlog.write(traceback.format_exc() + "\n")

                row = {
                    "qid": x.get("qid", ""),
                    "question": x.get("question", ""),
                    "gold_answer": x.get("answer", ""),
                    "pred_answer": "",
                    "raw_pred_answer": None,
                    "steps": -1,
                    "state": {},
                    "retrieved_evidence": [],
                    "rationale": None,
                    "error": repr(e),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if i % 10 == 0:
                print(f"[2wiki] processed {i}/{len(rows)}  success={success} failed={failed}")

    print(f"[done] wrote predictions to {output_path}")
    print(f"[done] success={success} failed={failed}")
    print(f"[done] error log at {log_path}")

if __name__ == "__main__":
    app()
