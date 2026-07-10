from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
import typer
import orjson

app = typer.Typer()

def iter_json_records(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for x in obj:
                yield x
        elif isinstance(obj, dict):
            yield obj

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return " ".join(x.split())
    if isinstance(x, list):
        if all(isinstance(i, str) for i in x):
            return " ".join(" ".join(i.split()) for i in x)
        return " ".join(normalize_text(i) for i in x)
    if isinstance(x, dict):
        for k in ["text", "contents", "document", "paragraph", "paragraph_text"]:
            if k in x:
                return normalize_text(x[k])
    return str(x)

def add_doc(title: str, text: str, docs: dict[str, dict]):
    title = (title or "").strip()
    text = normalize_text(text)
    if not text:
        return
    key = hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()
    docs[key] = {
        "id": key,
        "title": title,
        "text": text,
    }

def extract_from_record(rec: dict, docs: dict[str, dict]):
    context = rec.get("context")
    if isinstance(context, list):
        for item in context:
            if isinstance(item, list) and len(item) >= 2:
                title = item[0]
                text = item[1]
                add_doc(title, text, docs)
            elif isinstance(item, dict):
                title = item.get("title", "")
                text = (
                    item.get("sentences")
                    or item.get("text")
                    or item.get("paragraphs")
                    or item.get("contents")
                    or ""
                )
                add_doc(title, text, docs)

    if any(k in rec for k in ["title", "text", "contents", "document"]):
        add_doc(
            rec.get("title", ""),
            rec.get("text") or rec.get("contents") or rec.get("document") or "",
            docs,
        )

    paragraphs = rec.get("paragraphs")
    if isinstance(paragraphs, list):
        title = rec.get("title", "")
        for p in paragraphs:
            if isinstance(p, dict):
                add_doc(title or p.get("title", ""), p.get("text") or p.get("contents") or "", docs)
            else:
                add_doc(title, p, docs)

@app.command()
def main(
    input_dir: Path = typer.Option(..., exists=True, help="Hotpot dataset root dir"),
    output: Path = typer.Option(..., help="Output corpus jsonl"),
):
    docs: dict[str, dict] = {}

    files = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}]
    )

    print(f"[info] found {len(files)} json/jsonl files under {input_dir}")
    for fp in files:
        try:
            count_before = len(docs)
            for rec in iter_json_records(fp):
                if isinstance(rec, dict):
                    extract_from_record(rec, docs)
            print(f"[ok] {fp} -> +{len(docs) - count_before} docs")
        except Exception as e:
            print(f"[skip] {fp}: {e}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for row in docs.values():
            f.write(orjson.dumps(row) + b"\n")

    print(f"[done] wrote {len(docs)} docs to {output}")

if __name__ == "__main__":
    app()
