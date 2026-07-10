from __future__ import annotations

import re
from typing import Any


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def _fuzzy_title_match(title: str, entity: str) -> bool:
    t = _norm(title)
    e = _norm(entity)
    if not t or not e:
        return False
    return t == e or t.startswith(e) or e.startswith(t) or e in t


def normalize_entity_like_value(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().rstrip(".")
    if not s:
        return None

    m = re.search(r"\bis\b\s+(.+)$", s, flags=re.I)
    if m:
        tail = m.group(1).strip().rstrip(".")
        if tail:
            return tail

    m = re.search(r"\bwas\b\s+(.+)$", s, flags=re.I)
    if m:
        tail = m.group(1).strip().rstrip(".")
        if tail:
            return tail

    return s


def build_evidence_map(all_evidence: list[Any]) -> dict[str, Any]:
    out = {}
    for e in all_evidence:
        ev_id = _get(e, "evidence_id")
        if ev_id:
            out[ev_id] = e
    return out


def _slot_map(state: Any) -> dict[str, Any]:
    out = {}
    for s in _get(state, "slots", []) or []:
        sid = _get(s, "slot_id")
        if sid:
            out[sid] = s
    return out


def has_dependency_page_support(slot: Any, slot_map: dict[str, Any], evidence_map: dict[str, Any]) -> bool:
    dep_ids = _get(slot, "depends_on", []) or []
    if not dep_ids:
        return True

    dep_values = []
    for dep in dep_ids:
        dep_slot = slot_map.get(dep)
        if dep_slot and _get(dep_slot, "status", "") == "confirmed" and _get(dep_slot, "value", None):
            v = normalize_entity_like_value(_get(dep_slot, "value"))
            if v:
                dep_values.append(v)

    if not dep_values:
        return False

    ev_ids = _get(slot, "evidence_ids", []) or []
    for ev_id in ev_ids:
        if isinstance(ev_id, str) and ev_id.startswith("s"):
            continue
        ev = evidence_map.get(ev_id)
        if ev is None:
            continue
        ev_title = _get(ev, "source_title", "") or ""
        for dep_value in dep_values:
            if _fuzzy_title_match(ev_title, dep_value):
                return True

    return False


def _looks_synthetic_boolean(slot: Any) -> bool:
    slot_type = _get(slot, "slot_type", "other")
    desc = str(_get(slot, "description", "")).lower()
    if slot_type != "boolean":
        return False
    if "which director was born" in desc or "which film has" in desc or "are both" in desc or "same country" in desc:
        return True
    return False


def _should_validate_support_path(slot: Any) -> bool:
    status = _get(slot, "status", "")
    role = _get(slot, "target_role", "other")
    slot_type = _get(slot, "slot_type", "other")
    dep_ids = _get(slot, "depends_on", []) or []

    if status != "confirmed":
        return False
    if not dep_ids:
        return False
    if role not in {"dependent", "answer_target"}:
        return False
    if _looks_synthetic_boolean(slot):
        return False

    # validate entity/relation/attribute/date dependent slots
    if slot_type in {"entity", "relation", "attribute", "date"}:
        return True

    return False


def _question_expects_boolean(question: str) -> bool:
    q = question.lower().strip()
    return q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can "))


def posthoc_validate_state(state: Any, all_evidence: list[Any]) -> Any:
    evidence_map = build_evidence_map(all_evidence)
    slot_map = _slot_map(state)
    question = _get(state, "question", "")

    for slot in _get(state, "slots", []) or []:
        if _get(slot, "value", None):
            new_v = normalize_entity_like_value(_get(slot, "value"))
            if new_v is not None:
                _set(slot, "value", new_v)

        # prevent WH questions from ending with boolean answer_target
        if _get(slot, "target_role", "") == "answer_target":
            if not _question_expects_boolean(question) and _get(slot, "slot_type", "") == "boolean":
                _set(slot, "status", "candidate")
                _set(slot, "confidence", min(float(_get(slot, "confidence", 0.0)), 0.35))

        if _should_validate_support_path(slot):
            if not has_dependency_page_support(slot, slot_map, evidence_map):
                _set(slot, "status", "candidate")
                conf = float(_get(slot, "confidence", 0.0))
                _set(slot, "confidence", min(conf, 0.45))
                note = _get(slot, "notes", "") or ""
                suffix = " [demoted: missing support-path from dependency entity page]"
                if suffix not in note:
                    _set(slot, "notes", note + suffix)

    rebuild = getattr(state, "rebuild_views", None)
    if callable(rebuild):
        rebuild()
    return state
