from __future__ import annotations

from pathlib import Path
from typing import Any, List
from jinja2 import Template


class TypedTargetSelector:
    """
    Conservative target selector:
    - only runs on entity-like answers
    - does not replace numeric/date canonicalization logic
    - can keep current raw answer unchanged if evidence is insufficient
    """

    def __init__(self, llm, prompt_path: str | Path = "prompts/select_answer.txt"):
        self.llm = llm
        self.template = Template(Path(prompt_path).read_text(encoding="utf-8"))

    def _norm(self, s: str) -> str:
        return " ".join(str(s or "").split()).strip()

    def _answer_target_slot(self, state: Any):
        for s in getattr(state, "slots", []) or []:
            if getattr(s, "target_role", "") == "answer_target":
                return s
        return None

    def _confirmed_facts(self, state: Any) -> List[str]:
        vals = []
        for s in getattr(state, "slots", []) or []:
            if getattr(s, "status", "") == "confirmed":
                v = getattr(s, "value", None)
                if v not in [None, ""]:
                    vals.append(f"{getattr(s, 'slot_id', '')}: {self._norm(v)}")
        return vals[:8]

    def _evidence_text(self, evidence: List[Any], k: int = 5) -> str:
        rows = []
        for e in (evidence or [])[:k]:
            title = getattr(e, "source_title", "")
            text = getattr(e, "text", "")
            rows.append(f"[{title}] {text}")
        return "\n".join(rows)

    def should_run(self, question: str, raw_answer: str, state: Any, typed_spec: Any = None) -> bool:
        raw = self._norm(raw_answer).lower()
        answer_type = str(getattr(typed_spec, "answer_type", "") or "entity")
        operator_type = str(getattr(typed_spec, "operator_type", "") or "")

        # don't touch boolean/comparison/number/date
        if answer_type in {"boolean", "number", "date"}:
            return False
        if operator_type in {"comparison", "boolean_conjunction"}:
            return False

        # run on uncertain / target-sensitive entity-like outputs
        if raw in {"", "none", "unknown", "unresolved", "missing", "no"}:
            return True

        target_slot = self._answer_target_slot(state)
        if target_slot is None:
            return False

        # only run when target slot is not confidently confirmed
        if getattr(target_slot, "status", "") != "confirmed":
            return True

        # or when raw answer differs from confirmed answer target
        target_val = self._norm(getattr(target_slot, "value", "") or "")
        if target_val and raw != target_val.lower():
            return True

        return False

    def select(
        self,
        question: str,
        raw_answer: str,
        state: Any,
        evidence: List[Any],
        typed_spec_text: str,
    ) -> str:
        target_slot = self._answer_target_slot(state)
        answer_target_desc = getattr(target_slot, "description", "") if target_slot else ""
        confirmed_facts = self._confirmed_facts(state)
        evidence_text = self._evidence_text(evidence)

        prompt = self.template.render(
            question=question,
            typed_spec_text=typed_spec_text,
            raw_answer=raw_answer,
            answer_target_desc=answer_target_desc,
            confirmed_facts="\n".join(confirmed_facts),
            evidence_text=evidence_text,
        )

        out = self.llm.generate_json(prompt)
        final_answer = self._norm(out.get("final_answer", raw_answer))
        return final_answer or raw_answer
