from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    import faiss
except Exception:
    faiss = None


def human_bytes(n: int | None):
    if n is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for fn in files:
            p = Path(root) / fn
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def count_lines(path: Path):
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def get_text_from_obj(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    for k in [
        "text", "contents", "content", "passage", "paragraph",
        "chunk", "body", "document", "context"
    ]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # fallback: concatenate string fields
    parts = []
    for v in obj.values():
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def sample_jsonl_stats(path: Path, max_rows: int = 10000):
    n = 0
    chars = []
    words = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if n >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = get_text_from_obj(obj)
                if txt:
                    chars.append(len(txt))
                    words.append(len(txt.split()))
                n += 1
    except Exception:
        return {
            "sampled_rows": 0,
            "avg_chars": None,
            "avg_words": None,
        }

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "sampled_rows": n,
        "avg_chars": avg(chars),
        "avg_words": avg(words),
    }


def read_faiss_ntotal(path: Path):
    if faiss is None:
        return None
    try:
        idx = faiss.read_index(str(path))
        return int(idx.ntotal)
    except Exception as e:
        return f"read_error: {e}"


def collect_one(dataset: str, index_dir: Path):
    files = [p for p in index_dir.rglob("*") if p.is_file()] if index_dir.exists() else []

    faiss_files = [
        p for p in files
        if p.suffix == ".faiss" or p.name.endswith(".index")
    ]
    jsonl_files = [p for p in files if p.suffix == ".jsonl"]
    json_files = [p for p in files if p.suffix == ".json"]
    npy_files = [p for p in files if p.suffix == ".npy"]

    faiss_infos = []
    ntotal_sum = 0
    ntotal_has_numeric = False
    faiss_size = 0

    for fp in faiss_files:
        size = fp.stat().st_size
        faiss_size += size
        ntotal = read_faiss_ntotal(fp)
        if isinstance(ntotal, int):
            ntotal_sum += ntotal
            ntotal_has_numeric = True
        faiss_infos.append({
            "path": str(fp),
            "size_bytes": size,
            "size_human": human_bytes(size),
            "ntotal": ntotal,
        })

    # 选择最可能的 chunk/meta 文件：优先 chunks / corpus / passages / meta
    candidate_jsonl = []
    for p in jsonl_files:
        lower = p.name.lower()
        score = 0
        for key in ["chunk", "corpus", "passage", "meta", "doc"]:
            if key in lower:
                score += 1
        candidate_jsonl.append((score, p.stat().st_size, p))
    candidate_jsonl.sort(reverse=True)

    chunk_file = candidate_jsonl[0][2] if candidate_jsonl else None
    chunk_lines = count_lines(chunk_file) if chunk_file else None
    chunk_stats = sample_jsonl_stats(chunk_file) if chunk_file else {}

    total_size = dir_size(index_dir)

    return {
        "dataset": dataset,
        "index_dir": str(index_dir),
        "index_dir_size_bytes": total_size,
        "index_dir_size_human": human_bytes(total_size),
        "faiss_files_count": len(faiss_files),
        "faiss_total_size_bytes": faiss_size,
        "faiss_total_size_human": human_bytes(faiss_size),
        "faiss_ntotal_sum": ntotal_sum if ntotal_has_numeric else None,
        "faiss_files": faiss_infos,
        "jsonl_files_count": len(jsonl_files),
        "json_files_count": len(json_files),
        "npy_files_count": len(npy_files),
        "all_files_count": len(files),
        "chunk_or_meta_file": str(chunk_file) if chunk_file else None,
        "chunk_or_meta_lines": chunk_lines,
        "sampled_rows_for_chunk_stats": chunk_stats.get("sampled_rows"),
        "avg_chunk_chars_sampled": chunk_stats.get("avg_chars"),
        "avg_chunk_words_sampled": chunk_stats.get("avg_words"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot-index", default="data/indexes/hotpot")
    ap.add_argument("--twowiki-index", default="data/indexes/2wiki")
    ap.add_argument("--musique-index", default="data/indexes/musique")
    ap.add_argument("--output-json", default="outputs/index_stats/index_stats_detailed.json")
    ap.add_argument("--output-md", default="outputs/index_stats/index_stats_detailed.md")
    args = ap.parse_args()

    configs = [
        ("HotpotQA", Path(args.hotpot_index)),
        ("2Wiki", Path(args.twowiki_index)),
        ("MuSiQue", Path(args.musique_index)),
    ]

    rows = [collect_one(ds, p) for ds, p in configs]

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("| Dataset | Index Dir Size | FAISS Size | FAISS ntotal | Chunk/Meta Lines | Avg Chunk Chars | Avg Chunk Words | Files |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)

        lines.append(
            f"| {r['dataset']} | "
            f"{r['index_dir_size_human']} | "
            f"{r['faiss_total_size_human']} | "
            f"{fmt(r['faiss_ntotal_sum'])} | "
            f"{fmt(r['chunk_or_meta_lines'])} | "
            f"{fmt(r['avg_chunk_chars_sampled'])} | "
            f"{fmt(r['avg_chunk_words_sampled'])} | "
            f"{r['all_files_count']} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_md}")
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
