from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Slot(BaseModel):
    slot_id: str
    description: str
    slot_type: str = "other"
    depends_on: List[str] = Field(default_factory=list)
    target_role: str = "other"   # bridge | dependent | answer_target | other
    value: Optional[str] = None
    status: str = "missing"
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)
    notes: str = ""


class Evidence(BaseModel):
    evidence_id: str
    text: str
    source_id: str
    score: float
    source_title: str = ""


class Conflict(BaseModel):
    slot_id: str
    reason: str


class KnowledgeState(BaseModel):
    question: str
    slots: List[Slot]
    confirmed_facts: List[str] = Field(default_factory=list)
    candidate_facts: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    step: int = 0

    def rebuild_views(self) -> None:
        self.confirmed_facts = [f"{s.slot_id}: {s.value}" for s in self.slots if s.status == "confirmed" and s.value]
        self.candidate_facts = [f"{s.slot_id}: {s.value}" for s in self.slots if s.status == "candidate" and s.value]
        self.missing_slots = [s.slot_id for s in self.slots if s.status in {"missing", "candidate", "conflict"}]


class RunResult(BaseModel):
    question: str
    final_answer: str
    rationale: str
    steps: int
    state: KnowledgeState
    retrieved_evidence: List[Evidence]
