from __future__ import annotations

import json
from pathlib import Path
from typing import List
import faiss
import numpy as np
from csa_rag.schema import Evidence


def _norm_title(t: str) -> str:
    return " ".join(str(t or "").lower().split())


def _title_variants(t: str) -> list[str]:
    t = str(t or "").strip()
    out = [t]
    if "(" in t:
        out.append(t.split("(", 1)[0].strip())
    if "," in t:
        out.append(t.split(",", 1)[0].strip())
    return [x for x in out if x]


class DenseRetriever:
    def __init__(self, index_dir: Path, embedder):
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.embedder = embedder
        self.rows = []
        self.title_to_rows = {}

        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.rows.append(row)
                key = _norm_title(row.get("title", ""))
                if key:
                    self.title_to_rows.setdefault(key, []).append(row)

        for key in self.title_to_rows:
            self.title_to_rows[key].sort(key=lambda r: len(r.get("text", "")), reverse=True)

    def search(self, query: str, top_k: int = 10) -> List[Evidence]:
        q = self.embedder.encode([query])
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self.rows[idx]
            results.append(
                Evidence(
                    evidence_id=row["chunk_id"],
                    text=row["text"],
                    source_id=row["doc_id"],
                    score=float(score),
                    source_title=row.get("title", ""),
                )
            )
        return results

    def fetch_by_titles(self, titles: List[str], top_per_title: int = 1) -> List[Evidence]:
        seen = set()
        out: List[Evidence] = []
        for title in titles:
            key = _norm_title(title)
            rows = self.title_to_rows.get(key, [])
            for row in rows[:top_per_title]:
                chunk_id = row["chunk_id"]
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                out.append(
                    Evidence(
                        evidence_id=chunk_id,
                        text=row["text"],
                        source_id=row["doc_id"],
                        score=0.0,
                        source_title=row.get("title", ""),
                    )
                )
        return out

    def fetch_by_titles_fuzzy(self, titles: List[str], top_per_title: int = 1, max_candidates: int = 20) -> List[Evidence]:
        """
        Compatibility helper for pipeline versions that want fuzzy title hops.
        Conservative behavior:
        1) exact title match first
        2) if no exact hit, allow prefix/containment fallback on normalized titles
        """
        seen = set()
        out: List[Evidence] = []

        for title in titles:
            candidates = []

            for variant in _title_variants(title):
                key = _norm_title(variant)

                # exact
                rows = self.title_to_rows.get(key, [])
                if rows:
                    candidates.extend(rows[:top_per_title])

                # fallback
                if not rows:
                    matched_keys = []
                    for k in self.title_to_rows.keys():
                        if k.startswith(key) or key.startswith(k) or key in k:
                            matched_keys.append(k)
                            if len(matched_keys) >= max_candidates:
                                break
                    for mk in matched_keys:
                        candidates.extend(self.title_to_rows[mk][:top_per_title])

            for row in candidates:
                chunk_id = row["chunk_id"]
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                out.append(
                    Evidence(
                        evidence_id=chunk_id,
                        text=row["text"],
                        source_id=row["doc_id"],
                        score=0.0,
                        source_title=row.get("title", ""),
                    )
                )

        return out
