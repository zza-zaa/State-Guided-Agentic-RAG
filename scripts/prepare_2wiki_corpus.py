from __future__ import annotations
import json
import hashlib
from pathlib import Path
import typer
import orjson

app = typer.Typer()

def add_doc(title: str, text: str, docs: dict):
    title = (title or "").strip()
    text = " ".join(str(text).split())
    if not text:
        return
    key = hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()
    docs[key] = {"id": key, "title": title, "text": text}

def iter_records(path: Path):
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for x in obj:
                yield x
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

@app.command()
def main(input_dir: Path = typer.Option(..., exists=True), output: Path = typer.Option(...)):
    docs = {}
    files = sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}])
    print(f"[info] found {len(files)} files")
    for fp in files:
        try:
            before = len(docs)
            for rec in iter_records(fp):
                if not isinstance(rec, dict):
                    continue
                context = rec.get("context")
                if isinstance(context, list):
                    for item in context:
                        if isinstance(item, list) and len(item) >= 2:
                            add_doc(item[0], " ".join(item[1]) if isinstance(item[1], list) else item[1], docs)
                        elif isinstance(item, dict):
                            add_doc(item.get("title", ""), item.get("text", "") or item.get("contents", ""), docs)
                if "title" in rec and ("text" in rec or "contents" in rec):
                    add_doc(rec.get("title", ""), rec.get("text", "") or rec.get("contents", ""), docs)
            print(f"[ok] {fp} -> +{len(docs)-before} docs")
        except Exception as e:
            print(f"[skip] {fp}: {e}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for row in docs.values():
            f.write(orjson.dumps(row) + b"\n")
    print(f"[done] wrote {len(docs)} docs to {output}")

if __name__ == "__main__":
    app()
