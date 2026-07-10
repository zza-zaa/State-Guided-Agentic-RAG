from __future__ import annotations

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

from csa_rag.config import load_yaml
from csa_rag.llm.factory import build_llm
from csa_rag.state.decomposer import EvidenceSlotDecomposer
from dataset_utils import load_json_or_jsonl, normalize_hotpot_record, normalize_2wiki_record, normalize_musique_record

app = typer.Typer()

@app.command()
def main(
    dataset_name: str = typer.Option(..., help="hotpot|2wiki|musique"),
    dataset_path: Path = typer.Option(..., exists=True),
    models_config: Path = typer.Option(..., exists=True),
    limit: int = typer.Option(10),
    offset: int = typer.Option(0),
):
    cfg = load_yaml(models_config)
    llm = build_llm(cfg)
    decomposer = EvidenceSlotDecomposer(llm)

    rows = list(load_json_or_jsonl(dataset_path))[offset: offset + limit]

    normalizer = {
        "hotpot": normalize_hotpot_record,
        "2wiki": normalize_2wiki_record,
        "musique": normalize_musique_record,
    }[dataset_name]

    for i, rec in enumerate(rows, 1):
        x = normalizer(rec)
        q = x.get("question", "")
        print("=" * 140)
        print(f"[{i}] qid={x.get('qid','')}")
        print("question:", q)
        state = decomposer.decompose(q)
        if hasattr(state, "model_dump_json"):
            print(state.model_dump_json(indent=2))
        else:
            print(state)

if __name__ == "__main__":
    app()
