from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import typer
from csa_rag.embedding.bge_embedder import BGEEmbedder
from csa_rag.retrieval.chunker import Chunker
from csa_rag.retrieval.indexer import build_faiss_index_from_jsonl

app = typer.Typer()


@app.command()
def main(
    input: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
    model_name: str = typer.Option("BAAI/bge-m3"),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    embedder = BGEEmbedder(model_name=model_name)
    chunker = Chunker(min_chars=300, max_chars=1200, overlap=120)
    build_faiss_index_from_jsonl(input_path=input, output_dir=output, chunker=chunker, embedder=embedder)
    typer.echo(f"[done] index written to {output}")


if __name__ == "__main__":
    app()
