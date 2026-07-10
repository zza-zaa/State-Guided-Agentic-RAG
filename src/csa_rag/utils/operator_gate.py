from __future__ import annotations

import re
from typing import Any


def _quoted_entities(text: str) -> list[str]:
    out = []
    for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", str(text)):
        x = a or b
        x = x.strip()
        if x:
            out.append(x)
    return out


def infer_operator_family(question: str) -> str:
    q = question.lower().strip()

    comparison_markers = [
        "older", "younger",
        "born first", "born later",
        "died earlier", "died later",
        "released earlier", "released later",
        "earlier", "later",
    ]
    if any(m in q for m in comparison_markers):
        return "comparison"

    if "same country" in q or "same city" in q or "same nationality" in q:
        return "boolean_equivalence"

    return "none"


def infer_output_domain(question: str, answer_target_desc: str = "") -> str:
    q = f"{question} {answer_target_desc}".lower()

    if q.startswith("which film") or " which film" in q:
        return "film"
    if q.startswith("which school") or " which school" in q:
        return "school"
    if q.startswith("which team") or " what team" in q:
        return "team"
    if q.startswith("which song") or " which song" in q:
        return "song"
    if q.startswith("which album") or " which album" in q:
        return "album"
    if q.startswith("who ") or "who is" in q or "who was" in q:
        return "person"
    if q.startswith("when "):
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith("are ") or q.startswith("is "):
        return "boolean"

    return "other"


def _answer_target(state: dict[str, Any]) -> dict[str, Any] | None:
    for s in state.get("slots", []) or []:
        if s.get("target_role") == "answer_target":
            return s
    return None


def _slot_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s.get("slot_id"): s for s in state.get("slots", []) or [] if s.get("slot_id")}


def candidate_set_from_question(question: str, state: dict[str, Any]) -> list[str]:
    family = infer_operator_family(question)
    target = _answer_target(state) or {}
    domain = infer_output_domain(question, target.get("description", ""))

    if family != "comparison":
        return []

    candidates = []

    # 1) explicit quoted entities in question
    candidates.extend(_quoted_entities(question))

    # 2) bridge slot values
    for s in state.get("slots", []) or []:
        if s.get("target_role") == "bridge" and s.get("status") == "confirmed" and s.get("value"):
            candidates.append(str(s["value"]).strip())

    # 3) quoted entities in bridge descriptions / dependency descriptions
    for s in state.get("slots", []) or []:
        candidates.extend(_quoted_entities(s.get("description", "")))

    # unique preserve order
    seen = set()
    uniq = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        uniq.append(c)

    # For comparison questions, usually exactly two core candidates matter.
    # Keep first several rather than hard failing when not quoted.
    if domain in {"film", "song", "album", "school", "team", "person"}:
        return uniq[:6]

    return uniq[:6]


def should_use_operator_answer(
    question: str,
    state: dict[str, Any],
    operator_answer: str | None,
    llm_answer: str | None,
) -> bool:
    if not operator_answer:
        return False

    family = infer_operator_family(question)
    if family == "none":
        return False

    if state.get("conflicts"):
        return False

    op = str(operator_answer).strip()
    if op == "":
        return False

    if family == "boolean_equivalence":
        return op.lower() in {"yes", "no"}

    if family == "comparison":
        candidates = candidate_set_from_question(question, state)
        if candidates:
            return op in candidates
        return False

    return False
