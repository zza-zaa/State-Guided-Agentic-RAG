from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import faiss
import numpy as np
import orjson
from csa_rag.retrieval.chunker import Chunker


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_faiss_index_from_jsonl(input_path: Path, output_dir: Path, chunker: Chunker, embedder: Any) -> None:
    rows = []
    texts = []
    for record in read_jsonl(input_path):
        doc_id = str(record.get("id", len(rows)))
        text = record.get("text") or record.get("contents") or record.get("document") or ""
        title = record.get("title", "")
        for i, chunk in enumerate(chunker.split(text)):
            chunk_id = f"{doc_id}::{i}"
            rows.append({"chunk_id": chunk_id, "doc_id": doc_id, "title": title, "text": chunk})
            texts.append(chunk)

    vecs = embedder.encode(texts)
    dim = int(vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with (output_dir / "chunks.jsonl").open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")
