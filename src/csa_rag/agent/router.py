from __future__ import annotations

from dataclasses import dataclass
from csa_rag.schema import KnowledgeState, Slot


@dataclass
class RouteDecision:
    action: str
    target_slot_id: str | None
    reason: str


class UncertaintyGuidedRouter:
    def _slot_map(self, state: KnowledgeState) -> dict[str, Slot]:
        return {s.slot_id: s for s in state.slots}

    def _deps_ready(self, slot: Slot, state: KnowledgeState) -> bool:
        if not slot.depends_on:
            return True
        smap = self._slot_map(state)
        for dep in slot.depends_on:
            dep_slot = smap.get(dep)
            if dep_slot is None:
                return False
            if dep_slot.status != "confirmed" or not dep_slot.value:
                return False
        return True

    def decide(self, state: KnowledgeState) -> RouteDecision:
        unresolved = [s for s in state.slots if s.status in {"missing", "candidate", "conflict"}]
        if not unresolved:
            return RouteDecision(action="answer", target_slot_id=None, reason="all slots resolved")

        def sort_key(slot: Slot):
            deps_ready = self._deps_ready(slot, state)
            is_conflict = 0 if slot.status == "conflict" else 1
            is_second_hop_ready = 0 if (slot.depends_on and deps_ready) else 1
            return (
                0 if deps_ready else 1,
                is_conflict,
                is_second_hop_ready,
                slot.confidence,
                -len(slot.depends_on),
            )

        slot = sorted(unresolved, key=sort_key)[0]

        if slot.slot_type == "entity":
            return RouteDecision(action="entity_search", target_slot_id=slot.slot_id, reason="entity slot unresolved")
        if slot.slot_type == "relation":
            return RouteDecision(action="relation_search", target_slot_id=slot.slot_id, reason="relation slot unresolved")
        return RouteDecision(action="semantic_search", target_slot_id=slot.slot_id, reason="generic unresolved slot")
