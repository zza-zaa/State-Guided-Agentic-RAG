from __future__ import annotations

from typing import Any, Iterable, List


class TypedAnswerGrounder:
    """
    Conservative typed answer grounding:
    - do NOT mine arbitrary evidence texts for numbers/dates/years
    - do NOT override comparison/boolean outputs
    - only use confirmed answer_target slot as a soft canonicalization source
    - only apply low-risk normalization
    """

    def _norm(self, s: str) -> str:
        return " ".join(str(s or "").split()).strip()

    def _lower(self, s: str) -> str:
        return self._norm(s).lower()

    def _normalize_yes_no(self, s: str) -> str:
        x = self._lower(s)
        if x in {"true", "yes", "y"}:
            return "yes"
        if x in {"false", "no", "n"}:
            return "no"
        return self._norm(s)

    def _strip_leading_article(self, s: str) -> str:
        s = self._norm(s)
        for prefix in ["The ", "the ", "\"The ", "'The "]:
            if s.startswith(prefix):
                return s[len(prefix):]
        return s

    def _strip_trailing_country(self, s: str) -> str:
        s = self._norm(s)
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 3:
            return ", ".join(parts[:-1])
        return s

    def _answer_target_slot(self, state: Any):
        for s in getattr(state, "slots", []) or []:
            if getattr(s, "target_role", "") == "answer_target":
                return s
        return None

    def _confirmed_answer_target_value(self, state: Any) -> str:
        slot = self._answer_target_slot(state)
        if slot is None:
            return ""
        if getattr(slot, "status", "") != "confirmed":
            return ""
        value = getattr(slot, "value", None)
        if value in [None, ""]:
            return ""
        return self._norm(str(value))

    def _prefer_more_complete_person(self, raw: str, target: str) -> str:
        raw_n = self._norm(raw)
        tgt_n = self._norm(target)
        if not raw_n:
            return tgt_n
        if not tgt_n:
            return raw_n
        rl = raw_n.lower()
        tl = tgt_n.lower()

        if rl in tl and len(tgt_n) > len(raw_n):
            return tgt_n
        if tl in rl and len(raw_n) >= len(tgt_n):
            return raw_n
        return raw_n

    def _equivalent_up_to_article(self, a: str, b: str) -> bool:
        return self._lower(self._strip_leading_article(a)) == self._lower(self._strip_leading_article(b))

    def ground(
        self,
        question: str,
        raw_answer: str,
        state: Any,
        evidence: List[Any],
        typed_spec: Any = None,
    ) -> str:
        question_l = self._lower(question)
        raw = self._norm(raw_answer)

        relation_family = str(getattr(typed_spec, "relation_family", "") or "")
        operator_type = str(getattr(typed_spec, "operator_type", "") or "")
        answer_type = str(getattr(typed_spec, "answer_type", "") or "entity")
        granularity = str(getattr(typed_spec, "answer_granularity", "") or "default")

        # 0. leave comparisons / booleans alone except simple normalization
        if operator_type in {"comparison", "boolean_conjunction"} or answer_type == "boolean":
            return self._normalize_yes_no(raw)

        # 1. never try to "extract" numbers/dates/positions from arbitrary evidence here
        if answer_type in {
            "number", "date", "position", "award", "league",
            "record_label", "instrument", "organization", "cause_of_death"
        }:
            return raw

        target = self._confirmed_answer_target_value(state)

        # 2. nickname cleanup
        if "nickname" in question_l:
            cand = target or raw
            return self._strip_leading_article(cand)

        # 3. safe location cleanup only when raw/target are clearly related
        if relation_family == "location_admin" or answer_type == "location":
            raw2 = raw
            tgt2 = target

            if granularity in {"birthplace_locality", "admin_entity_exact"} or "hail from" in question_l:
                raw2 = self._strip_trailing_country(raw2)
                tgt2 = self._strip_trailing_country(tgt2)

            if raw2 and tgt2:
                rl = raw2.lower()
                tl = tgt2.lower()
                if rl in tl:
                    return raw2
                if tl in rl:
                    return tgt2
            return raw2 or tgt2 or raw

        # 4. safe person canonicalization
        if answer_type == "person":
            if target:
                return self._prefer_more_complete_person(raw, target)
            return raw

        # 5. generic article-only normalization when clearly equivalent
        if raw and target and self._equivalent_up_to_article(raw, target):
            return self._strip_leading_article(target)

        # 6. if raw is empty/unknown but confirmed target exists, use it
        if self._lower(raw) in {"", "none", "unknown", "unresolved", "missing"} and target:
            return target

        return raw
