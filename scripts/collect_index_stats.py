from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import faiss
except Exception:
    faiss = None


def human_bytes(n):
    if n is None:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024


def count_lines(path: Path):
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def dir_size(path: Path):
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except Exception:
                pass
    return total


def find_files(index_dir: Path):
    files = []
    for p in index_dir.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--hotpot-index", default="data/indexes/hotpot")
    ap.add_argument("--twowiki-index", default="data/indexes/2wiki")
    ap.add_argument("--musique-index", default="data/indexes/musique")
    args = ap.parse_args()

    datasets = {
        "hotpot": Path(args.hotpot_index),
        "2wiki": Path(args.twowiki_index),
        "musique": Path(args.musique_index),
    }

    rows = []

    for name, d in datasets.items():
        all_files = find_files(d)
        faiss_files = [p for p in all_files if p.suffix == ".faiss" or p.name.endswith(".index")]
        jsonl_files = [p for p in all_files if p.suffix == ".jsonl"]
        json_files = [p for p in all_files if p.suffix == ".json"]
        npy_files = [p for p in all_files if p.suffix == ".npy"]

        ntotal = None
        faiss_size = 0
        faiss_paths = []
        if faiss_files:
            for fp in faiss_files:
                faiss_size += fp.stat().st_size
                faiss_paths.append(str(fp))
            if faiss is not None:
                try:
                    idx = faiss.read_index(str(faiss_files[0]))
                    ntotal = int(idx.ntotal)
                except Exception as e:
                    ntotal = f"read_error: {e}"

        chunk_like = []
        for p in jsonl_files:
            lower = p.name.lower()
            if any(k in lower for k in ["chunk", "meta", "corpus", "passage", "doc"]):
                chunk_like.append(p)

        chunk_lines = None
        chunk_file = None
        if chunk_like:
            chunk_file = str(chunk_like[0])
            chunk_lines = count_lines(chunk_like[0])

        rows.append({
            "dataset": name,
            "index_dir": str(d),
            "dir_size_bytes": dir_size(d),
            "dir_size_human": human_bytes(dir_size(d)),
            "faiss_files": faiss_paths,
            "faiss_size_bytes": faiss_size,
            "faiss_size_human": human_bytes(faiss_size),
            "faiss_ntotal_first_index": ntotal,
            "jsonl_files_count": len(jsonl_files),
            "json_files_count": len(json_files),
            "npy_files_count": len(npy_files),
            "chunk_or_meta_file": chunk_file,
            "chunk_or_meta_lines": chunk_lines,
            "all_files_count": len(all_files),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| Dataset | Index Dir Size | FAISS Size | FAISS ntotal | Chunk/Meta Lines | Files |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['dataset']} | {r['dir_size_human']} | {r['faiss_size_human']} | "
            f"{r['faiss_ntotal_first_index']} | {r['chunk_or_meta_lines']} | {r['all_files_count']} |"
        )


if __name__ == "__main__":
    main()
