from __future__ import annotations

import re
from typing import Any, List


class EvidencePathScorer:
    def _norm(self, s: str) -> str:
        return " ".join(str(s or "").split()).strip()

    def _lower(self, s: str) -> str:
        return self._norm(s).lower()

    def _answer_target_slot(self, state: Any):
        for s in getattr(state, "slots", []) or []:
            if getattr(s, "target_role", "") == "answer_target":
                return s
        return None

    def _confirmed_values(self, state: Any) -> List[str]:
        vals = []
        for s in getattr(state, "slots", []) or []:
            if getattr(s, "status", "") == "confirmed":
                v = getattr(s, "value", None)
                if v not in [None, ""]:
                    vals.append(self._norm(str(v)))
        return vals

    def _slot_keywords(self, slot: Any) -> List[str]:
        desc = self._lower(getattr(slot, "description", "") or "")
        toks = re.findall(r"[a-zA-Z][a-zA-Z\-]+", desc)
        stop = {
            "what","which","who","when","where","the","a","an","of","for","in",
            "to","is","was","are","did","does","and","or","by","from","on",
            "that","this","with","during","other"
        }
        return [t for t in toks if t not in stop and len(t) > 2]

    def _typed_bonus(self, text_l: str, typed_spec: Any) -> float:
        relation_family = str(getattr(typed_spec, "relation_family", "") or "")
        answer_type = str(getattr(typed_spec, "answer_type", "") or "")
        bonus = 0.0

        if relation_family == "kinship":
            if any(t in text_l for t in ["father", "mother", "son", "daughter", "grandfather", "grandmother", "spouse"]):
                bonus += 1.0

        if relation_family == "location_admin":
            if any(t in text_l for t in ["city", "county", "province", "state", "country", "located", "headquartered"]):
                bonus += 1.0

        if relation_family == "entity_attribute":
            attr_cues = {
                "position": ["position", "centre", "center", "goalkeeper", "midfielder", "forward"],
                "award": ["award", "won", "recipient"],
                "league": ["league", "conference"],
                "organization": ["company", "organization", "owned", "founded"],
                "country_or_nationality": ["country", "nationality", "english", "american", "british"],
                "person": ["born", "writer", "director", "producer", "composer", "actor", "singer"],
                "number": ["population", "copies", "seated", "capacity", "million", "thousand"],
            }
            cues = attr_cues.get(answer_type, [])
            if any(t in text_l for t in cues):
                bonus += 1.0

        return bonus

    def score(self, ev: Any, state: Any, slot: Any, typed_spec: Any = None, rank_idx: int = 0) -> float:
        score = 0.0

        text = self._norm(getattr(ev, "text", "") or "")
        title = self._norm(getattr(ev, "source_title", "") or "")
        text_l = self._lower(text)
        title_l = self._lower(title)

        # keep original order as weak prior
        score += max(0.0, 1.5 - 0.08 * rank_idx)

        # answer-target alignment
        target_slot = self._answer_target_slot(state)
        if target_slot is not None:
            target_desc = self._lower(getattr(target_slot, "description", "") or "")
            for kw in re.findall(r"[a-zA-Z][a-zA-Z\-]+", target_desc):
                if len(kw) > 3 and kw in text_l:
                    score += 0.35

        # current slot keyword overlap
        for kw in self._slot_keywords(slot):
            if kw in text_l:
                score += 0.5
            if kw in title_l:
                score += 0.3

        # confirmed value / bridge consistency
        for val in self._confirmed_values(state):
            vl = self._lower(val)
            if vl and vl in text_l:
                score += 0.8
            if vl and vl in title_l:
                score += 0.6

        # typed hints
        if typed_spec is not None:
            score += self._typed_bonus(text_l, typed_spec)

        # discourage obviously unhelpful snippets
        if any(bad in text_l for bad in ["disambiguation", "may refer to", "unknown", "unresolved"]):
            score -= 0.8

        return score

    def select(self, evidence: List[Any], state: Any, slot: Any, typed_spec: Any = None, limit: int = 8) -> List[Any]:
        scored = []
        for i, ev in enumerate(evidence or []):
            s = self.score(ev, state=state, slot=slot, typed_spec=typed_spec, rank_idx=i)
            scored.append((s, i, ev))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [ev for _, _, ev in scored[:limit]]
