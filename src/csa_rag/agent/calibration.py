from __future__ import annotations

from csa_rag.schema import KnowledgeState
from csa_rag.utils.question_constraints import is_invalid_answer_for_question


class StateCalibrator:
    def __init__(self, stop_confidence: float = 0.83):
        self.stop_confidence = stop_confidence

    def should_stop(self, state: KnowledgeState, max_steps: int) -> bool:
        if state.step >= max_steps:
            return True

        if any(slot.status == "conflict" for slot in state.slots):
            return False

        answer_targets = [s for s in state.slots if getattr(s, "target_role", "other") == "answer_target"]

        if not answer_targets:
            if any(slot.status == "missing" for slot in state.slots):
                return False
            if any(slot.confidence < self.stop_confidence for slot in state.slots):
                return False
            return True

        # 所有 answer_target 必须 confirmed 且值合法
        for s in answer_targets:
            if s.status != "confirmed":
                return False
            if s.confidence < self.stop_confidence:
                return False
            if is_invalid_answer_for_question(state.question, s.value):
                return False
            if not s.evidence_ids:
                return False

        return True
