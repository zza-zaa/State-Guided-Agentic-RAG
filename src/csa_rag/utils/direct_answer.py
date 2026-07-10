from __future__ import annotations

import re
from typing import Any


MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _parse_date(text: str) -> tuple[int, int, int] | None:
    s = str(text).strip()

    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return (int(m.group(1)), 1, 1)

    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        d = int(m.group(1))
        mon = MONTHS.get(m.group(2).lower())
        y = int(m.group(3))
        if mon:
            return (y, mon, d)

    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        mon = MONTHS.get(m.group(1).lower())
        d = int(m.group(2))
        y = int(m.group(3))
        if mon:
            return (y, mon, d)

    return None


def _slot_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s.get("slot_id"): s for s in state.get("slots", []) or [] if s.get("slot_id")}


def _answer_target(state: dict[str, Any]) -> dict[str, Any] | None:
    for s in state.get("slots", []) or []:
        if s.get("target_role") == "answer_target":
            return s
    return None


def _quoted_texts(desc: str) -> list[str]:
    return [x.strip() for x in re.findall(r"'([^']+)'|\"([^\"]+)\"", str(desc)) for x in x if x.strip()]


def _first_quoted(desc: str) -> str | None:
    xs = _quoted_texts(desc)
    return xs[0] if xs else None


def _extract_subject_from_description(desc: str) -> str | None:
    desc = str(desc).strip()

    patterns = [
        r"^(.+?)'s birth date",
        r"^(.+?)'s death date",
        r"^(.+?)'s release date",
        r"^(.+?)'s spouse",
        r"^(.+?)'s father",
        r"^(.+?)'s mother",
        r"^(.+?)'s child",
        r"^(.+?)'s sibling",
        r"^The country where (.+?) is located",
        r"^The release date of the film '(.+?)'",
        r"^The director of '(.+?)'",
        r"^The director of the film '(.+?)'",
        r"^Identify the director of the film '(.+?)'",
    ]
    for p in patterns:
        m = re.match(p, desc, flags=re.I)
        if m:
            return m.group(1).strip()
    return None


def _infer_output_domain(question: str, answer_target_desc: str = "") -> str:
    q = f"{question} {answer_target_desc}".lower()

    # entity domains first
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
    if q.startswith("are ") or q.startswith("is ") or " same country" in q:
        return "boolean"

    return "other"


def _trace_output_label(slot: dict[str, Any], slot_map: dict[str, dict[str, Any]], output_domain: str) -> str | None:
    """
    For comparison tasks, return the entity in the domain requested by the question.
    Example:
      - compare directors' birth years, but output domain is film => return film title, not director name
      - compare people's birth years, output domain person => return person name
    """

    # If the target domain is film/song/album/school/team, prefer quoted entity from upstream bridge description.
    if output_domain in {"film", "song", "album", "school", "team"}:
        dep_ids = slot.get("depends_on", []) or []
        if dep_ids:
            dep = slot_map.get(dep_ids[0])
            if dep:
                q = _first_quoted(dep.get("description", ""))
                if q:
                    return q
                # fallback: sometimes bridge description subject itself is the target domain
                subj = _extract_subject_from_description(dep.get("description", ""))
                if subj:
                    return subj

    # For person domain, prefer the subject of this slot itself, then upstream bridge value
    if output_domain == "person":
        subj = _extract_subject_from_description(slot.get("description", ""))
        if subj:
            return subj
        dep_ids = slot.get("depends_on", []) or []
        if dep_ids:
            dep = slot_map.get(dep_ids[0])
            if dep and dep.get("value"):
                return str(dep.get("value")).strip()

    # Generic fallback: upstream quoted entity, then upstream value, then own subject
    dep_ids = slot.get("depends_on", []) or []
    if dep_ids:
        dep = slot_map.get(dep_ids[0])
        if dep:
            q = _first_quoted(dep.get("description", ""))
            if q:
                return q
            if dep.get("value"):
                return str(dep.get("value")).strip()

    subj = _extract_subject_from_description(slot.get("description", ""))
    if subj:
        return subj

    return None


def _bool_from_state(question: str, state: dict[str, Any]) -> str | None:
    q = question.lower().strip()
    slots = state.get("slots", []) or []

    if "same country" in q:
        vals = [
            str(s.get("value", "")).strip().lower()
            for s in slots
            if s.get("status") == "confirmed" and s.get("slot_type") == "location" and s.get("value")
        ]
        if len(vals) >= 2:
            return "yes" if vals[0] == vals[1] else "no"

    return None


def _comparison_from_state(question: str, state: dict[str, Any]) -> str | None:
    q = question.lower().strip()
    slot_map = _slot_map(state)
    answer_target = _answer_target(state)
    answer_target_desc = answer_target.get("description", "") if answer_target else ""
    output_domain = _infer_output_domain(question, answer_target_desc)

    # direct operator only for clear comparison questions
    comparison_keywords = [
        "older", "younger", "born first", "born later",
        "released earlier", "released later",
        "died earlier", "died later",
        "earlier", "later",
    ]
    if not any(k in q for k in comparison_keywords):
        return None

    comparable = []
    for s in state.get("slots", []) or []:
        if s.get("status") != "confirmed" or not s.get("value"):
            continue
        dt = _parse_date(str(s.get("value")))
        if dt is not None:
            label = _trace_output_label(s, slot_map, output_domain)
            if label:
                comparable.append((s, dt, label))

    if len(comparable) < 2:
        return None

    # use first two date-like slots; if labels collapse to same item, abort and fallback
    (_, d_a, label_a), (_, d_b, label_b) = comparable[:2]
    if label_a == label_b:
        return None

    if "older" in q or "born first" in q or "released earlier" in q or "died earlier" in q or "earlier" in q:
        return label_a if d_a < d_b else label_b

    if "younger" in q or "born later" in q or "released later" in q or "died later" in q or "later" in q:
        return label_a if d_a > d_b else label_b

    return None


def direct_answer_from_state(question: str, state: dict[str, Any]) -> str | None:
    if not state:
        return None

    # If there are explicit conflicts, be conservative and fallback
    if state.get("conflicts"):
        return None

    ans = _bool_from_state(question, state)
    if ans is not None:
        return ans

    ans = _comparison_from_state(question, state)
    if ans is not None:
        return ans

    return None
