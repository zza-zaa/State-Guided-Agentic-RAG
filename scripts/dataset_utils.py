from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def iter_examples(obj: Any):
    if isinstance(obj, list):
        for x in obj:
            yield x
    elif isinstance(obj, dict):
        for k in ["data", "examples", "items", "questions"]:
            v = obj.get(k)
            if isinstance(v, list):
                for x in v:
                    yield x
                return
        yield obj


def load_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        for x in iter_examples(obj):
            yield x
    else:
        raise ValueError(f"Unsupported file type: {path}")


def pick_answer(rec: Dict[str, Any]) -> str:
    if "answer" in rec and rec["answer"] is not None:
        if isinstance(rec["answer"], list):
            return str(rec["answer"][0]) if rec["answer"] else ""
        return str(rec["answer"])

    if "answers" in rec and rec["answers"] is not None:
        if isinstance(rec["answers"], list):
            return str(rec["answers"][0]) if rec["answers"] else ""
        return str(rec["answers"])

    if "answer_aliases" in rec and rec["answer_aliases"] is not None:
        aliases = rec["answer_aliases"]
        if isinstance(aliases, list):
            return str(aliases[0]) if aliases else ""
        return str(aliases)

    if "gold_answer" in rec and rec["gold_answer"] is not None:
        return str(rec["gold_answer"])

    return ""


def normalize_hotpot_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "qid": rec.get("_id") or rec.get("id") or "",
        "question": rec.get("question", ""),
        "answer": pick_answer(rec),
    }


def normalize_2wiki_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "qid": rec.get("_id") or rec.get("id") or "",
        "question": rec.get("question", ""),
        "answer": pick_answer(rec),
    }


def normalize_musique_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "qid": rec.get("id") or rec.get("_id") or "",
        "question": rec.get("question", ""),
        "answer": pick_answer(rec),
    }
