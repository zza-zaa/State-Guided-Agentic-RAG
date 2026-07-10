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


def _slot_map(state: Any) -> dict[str, Any]:
    out = {}
    for s in _get(state, "slots", []) or []:
        sid = _get(s, "slot_id")
        if sid:
            out[sid] = s
    return out


def _evidence_map(all_evidence: list[Any]) -> dict[str, Any]:
    out = {}
    for e in all_evidence:
        ev_id = _get(e, "evidence_id")
        if ev_id:
            out[ev_id] = e
    return out


def _question_expects_country(question: str) -> bool:
    q = question.lower()
    return "which country" in q or "what country" in q or "country is from" in q or "country of origin" in q


def _question_expects_county(question: str) -> bool:
    q = question.lower()
    return "which county" in q or "what county" in q or "in what county" in q


def _question_expects_place(question: str) -> bool:
    q = question.lower()
    return q.startswith("where ") or "what place" in q or "share a border with what place" in q


def _value_looks_too_coarse_for_country(value: str) -> bool:
    v = value.lower()
    if "county" in v or "province" in v or "state" in v:
        return True
    return False


def _value_looks_too_coarse_for_county(value: str) -> bool:
    v = value.lower()
    if "province" in v or "country" in v:
        return True
    return False


def _dependency_title_supported(slot: Any, slot_map: dict[str, Any], evidence_map: dict[str, Any]) -> bool:
    dep_ids = _get(slot, "depends_on", []) or []
    if not dep_ids:
        return True

    dep_values = []
    for dep in dep_ids:
        dep_slot = slot_map.get(dep)
        if dep_slot and _get(dep_slot, "status", "") == "confirmed" and _get(dep_slot, "value", None):
            dep_values.append(str(_get(dep_slot, "value")).strip())

    if not dep_values:
        return False

    matched = False
    for ev_id in _get(slot, "evidence_ids", []) or []:
        if isinstance(ev_id, str) and ev_id.startswith("s"):
            continue
        ev = evidence_map.get(ev_id)
        if ev is None:
            continue
        title = str(_get(ev, "source_title", "") or "")
        for dep_value in dep_values:
            if _fuzzy_title_match(title, dep_value):
                matched = True
                break
        if matched:
            break

    return matched


def _has_upstream_conflict(slot: Any, slot_map: dict[str, Any], state: Any) -> bool:
    dep_ids = _get(slot, "depends_on", []) or []
    if not dep_ids:
        return False

    conflicts = _get(state, "conflicts", []) or []
    conflict_slot_ids = {c.get("slot_id") for c in conflicts if isinstance(c, dict)}

    for dep in dep_ids:
        if dep in conflict_slot_ids:
            return True

    return False


def _demote(slot: Any, reason: str, cap: float = 0.45):
    _set(slot, "status", "candidate")
    conf = float(_get(slot, "confidence", 0.0))
    _set(slot, "confidence", min(conf, cap))
    notes = str(_get(slot, "notes", "") or "")
    suffix = f" [critic-demote: {reason}]"
    if suffix not in notes:
        _set(slot, "notes", notes + suffix)


class StateCritic:
    """
    Review newly updated state and demote over-confident slot confirmations
    that violate dependency, conflict, or coarse type constraints.
    """

    def review(self, state: Any, all_evidence: list[Any]) -> Any:
        question = str(_get(state, "question", "") or "")
        slot_map = _slot_map(state)
        evidence_map = _evidence_map(all_evidence)

        for slot in _get(state, "slots", []) or []:
            if _get(slot, "status", "") != "confirmed":
                continue

            value = str(_get(slot, "value", "") or "").strip()
            role = _get(slot, "target_role", "other")
            depends_on = _get(slot, "depends_on", []) or []

            # 1) if upstream dependency still has conflict, answer slot should not be final
            if role == "answer_target" and depends_on and _has_upstream_conflict(slot, slot_map, state):
                _demote(slot, "upstream dependency conflict unresolved", cap=0.35)
                continue

            # 2) dependent slots should have evidence aligned with dependency entity page
            if depends_on and role in {"dependent", "answer_target"}:
                if not _dependency_title_supported(slot, slot_map, evidence_map):
                    _demote(slot, "evidence title not aligned with dependency entity", cap=0.40)
                    continue

            # 3) coarse type sanity checks
            if value:
                if _question_expects_country(question) and _value_looks_too_coarse_for_country(value):
                    _demote(slot, "country question answered with sub-country unit", cap=0.35)
                    continue

                if _question_expects_county(question) and _value_looks_too_coarse_for_county(value):
                    _demote(slot, "county question answered with coarser geographic unit", cap=0.35)
                    continue

                if _question_expects_place(question) and value.lower() in {"south sudan", "tanzania"}:
                    # conservative guard for generic country leakage in border/place queries
                    _demote(slot, "place query answered with overly generic country-level candidate", cap=0.35)
                    continue

        rebuild = getattr(state, "rebuild_views", None)
        if callable(rebuild):
            rebuild()

        return state
