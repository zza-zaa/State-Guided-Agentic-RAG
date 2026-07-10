from __future__ import annotations

from pathlib import Path
from typing import List
from jinja2 import Template
from csa_rag.schema import Conflict, Evidence, KnowledgeState


class KnowledgeStateTracker:
    def __init__(self, llm, prompt_path: str | Path = "prompts/state_update.txt"):
        self.llm = llm
        self.template = Template(Path(prompt_path).read_text(encoding="utf-8"))

    def update(self, state: KnowledgeState, evidence: List[Evidence]) -> KnowledgeState:
        prompt = self.template.render(
            question=state.question,
            state_json=state.model_dump_json(indent=2),
            evidence_json=[e.model_dump() for e in evidence],
        )
        data = self.llm.generate_json(prompt)
        updates = {u["slot_id"]: u for u in data.get("slot_updates", [])}
        for slot in state.slots:
            if slot.slot_id in updates:
                u = updates[slot.slot_id]
                slot.value = u.get("value")
                slot.status = u.get("status", slot.status)
                slot.confidence = float(u.get("confidence", slot.confidence))
                slot.evidence_ids = u.get("supporting_evidence_ids", [])
                slot.notes = u.get("notes", "")
        state.conflicts = [Conflict(**c) for c in data.get("conflicts", [])]
        state.step += 1
        state.rebuild_views()
        return state
