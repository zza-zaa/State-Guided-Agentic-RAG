from __future__ import annotations

import json
import sys
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
from dataset_utils import load_json_or_jsonl, normalize_hotpot_record, normalize_2wiki_record, normalize_musique_record

app = typer.Typer()

@app.command()
def main(
    dataset_name: str = typer.Option(..., help="hotpot | 2wiki | musique"),
    dataset_path: Path = typer.Option(..., exists=True),
    index_dir: Path = typer.Option(..., exists=True),
    models_config: Path = typer.Option(..., exists=True),
    row_idx: int = typer.Option(0),
):
    if dataset_name not in {"hotpot", "2wiki", "musique"}:
        raise ValueError("dataset_name must be one of: hotpot, 2wiki, musique")

    rows = list(load_json_or_jsonl(dataset_path))
    rec = rows[row_idx]

    if dataset_name == "hotpot":
        x = normalize_hotpot_record(rec)
    elif dataset_name == "2wiki":
        x = normalize_2wiki_record(rec)
    else:
        x = normalize_musique_record(rec)

    print("qid:", x.get("qid"))
    print("question:", x.get("question"))
    print("gold:", x.get("answer"))

    pipe = CSARAGPipeline(models_config=models_config, index_dir=index_dir)
    result = pipe.run(x["question"])

    print("=" * 80)
    print("final_answer:", result.final_answer)
    print("steps:", result.steps)
    print("state:")
    print(result.state.model_dump_json(indent=2))
