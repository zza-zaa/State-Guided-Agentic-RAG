from __future__ import annotations

from pathlib import Path
from jinja2 import Template
from csa_rag.utils.relation_aware_hints import build_relation_aware_hint
from csa_rag.schema import KnowledgeState, Slot


class EvidenceSlotDecomposer:
    def __init__(self, llm, prompt_path: str | Path = "prompts/decompose.txt"):
        self.llm = llm
        self.template = Template(Path(prompt_path).read_text(encoding="utf-8"))

    def decompose(self, question: str, typed_spec_text: str = "") -> KnowledgeState:
        hint = build_relation_aware_hint(question)

        blocks = []
        if typed_spec_text:
            blocks.append(typed_spec_text)
        blocks.append(question)
        if hint:
            blocks.append(f"[Relation-aware planning hints]\n{hint}")

        aug_question = "\n\n".join(blocks)

        prompt = self.template.render(question=aug_question)
        data = self.llm.generate_json(prompt)
        slots = [Slot(**item) for item in data.get("slots", [])]
        state = KnowledgeState(question=question, slots=slots)
        state.rebuild_views()
        return state
