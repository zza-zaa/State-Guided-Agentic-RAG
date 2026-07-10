from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import typer
from rich import print
from csa_rag.agent.pipeline import CSARAGPipeline

app = typer.Typer()


@app.command()
def main(
    question: str = typer.Option(...),
    index_dir: Path = typer.Option(..., exists=True),
    models_config: Path = typer.Option(Path("configs/models.yaml"), exists=True),
) -> None:
    pipeline = CSARAGPipeline(models_config=models_config, index_dir=index_dir)
    result = pipeline.run(question)
    print(result.model_dump())


if __name__ == "__main__":
    app()
