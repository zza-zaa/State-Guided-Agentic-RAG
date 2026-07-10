from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class RetrievalPlan:
    strategy: str
    query: str
    restrict_titles: Optional[List[str]] = None
    topk_dense: int = 30
    topk_rerank: int = 10


class SlotRouter:
    """
    Soft typed integration version:
      - keep original routing backbone
      - use typed question spec only as soft bias
      - do not hard override state critic / final resolver
    """

    def route(self, state: Any, slot: Any, typed_spec: Any = None) -> RetrievalPlan:
        desc = str(getattr(slot, "description", "") or "")
        desc_l = desc.lower()
        depends_on = list(getattr(slot, "depends_on", []) or [])
        slot_type = str(getattr(slot, "slot_type", "other") or "other").lower()

        relation_family = str(getattr(typed_spec, "relation_family", "") or "")
        operator_type = str(getattr(typed_spec, "operator_type", "") or "")
        answer_type = str(getattr(typed_spec, "answer_type", "") or "")
        chain_depth_hint = int(getattr(typed_spec, "chain_depth_hint", 1) or 1)

        # 1) independent entity resolution
        if slot_type == "entity" and not depends_on:
            return RetrievalPlan(
                strategy="ENTITY_RESOLUTION",
                query=desc,
                restrict_titles=None,
                topk_dense=30 if chain_depth_hint <= 1 else 26,
                topk_rerank=10,
            )

        # 2) dependent attribute / relation
        if depends_on:
            parent_slots = [
                s for s in getattr(state, "slots", [])
                if getattr(s, "slot_id", None) in depends_on
                and getattr(s, "status", "") == "confirmed"
                and getattr(s, "value", None)
            ]
            if parent_slots:
                parent = parent_slots[0]
                parent_value = str(getattr(parent, "value", "")).strip()
                base_query = f"{parent_value} {desc}"

                if operator_type == "kinship_composition":
                    return RetrievalPlan(
                        strategy="KINSHIP_DEPENDENT",
                        query=base_query,
                        restrict_titles=[parent_value],
                        topk_dense=18,
                        topk_rerank=8,
                    )

                if relation_family == "location_admin":
                    return RetrievalPlan(
                        strategy="DEPENDENT_LOCATION_ATTRIBUTE",
                        query=base_query,
                        restrict_titles=[parent_value],
                        topk_dense=22,
                        topk_rerank=8,
                    )

                if relation_family == "entity_attribute":
                    typed_query = base_query
                    if answer_type in {
                        "position", "instrument", "award", "league",
                        "record_label", "country_or_nationality",
                        "organization", "person", "number"
                    }:
                        typed_query = f"{base_query} answer type: {answer_type}"
                    return RetrievalPlan(
                        strategy="DEPENDENT_TYPED_ATTRIBUTE",
                        query=typed_query,
                        restrict_titles=[parent_value],
                        topk_dense=24,
                        topk_rerank=10,
                    )

                return RetrievalPlan(
                    strategy="DEPENDENT_ATTRIBUTE",
                    query=base_query,
                    restrict_titles=[parent_value],
                    topk_dense=20,
                    topk_rerank=8,
                )

        # 3) typed comparison / boolean
        if operator_type == "comparison":
            return RetrievalPlan(
                strategy="COMPARISON_ATTRIBUTE",
                query=desc,
                restrict_titles=None,
                topk_dense=28,
                topk_rerank=10,
            )

        if operator_type == "boolean_conjunction":
            return RetrievalPlan(
                strategy="BOOLEAN_ATTRIBUTE",
                query=desc,
                restrict_titles=None,
                topk_dense=24,
                topk_rerank=8,
            )

        # 4) aggregation / location style
        if relation_family == "location_admin" or "located" in desc_l or "location" in desc_l or "where" in desc_l:
            return RetrievalPlan(
                strategy="AGGREGATION_LOCATION",
                query=desc,
                restrict_titles=None,
                topk_dense=24,
                topk_rerank=8,
            )

        # 5) typed generic entity attribute
        if relation_family == "entity_attribute":
            q = desc
            if answer_type not in {"entity", "boolean"}:
                q = f"{desc} answer type: {answer_type}"
            return RetrievalPlan(
                strategy="GENERIC_TYPED_ATTRIBUTE",
                query=q,
                restrict_titles=None,
                topk_dense=26,
                topk_rerank=10,
            )

        # 6) fallback
        return RetrievalPlan(
            strategy="GENERIC_FALLBACK",
            query=desc,
            restrict_titles=None,
            topk_dense=30,
            topk_rerank=10,
        )
