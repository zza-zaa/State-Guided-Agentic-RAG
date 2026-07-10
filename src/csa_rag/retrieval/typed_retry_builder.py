from __future__ import annotations

from typing import Any, List


class TypedRetryQueryBuilder:
    def build(
        self,
        question: str,
        slot: Any,
        confirmed_values: List[str],
        typed_spec: Any = None,
    ) -> str:
        desc = str(getattr(slot, "description", "") or "")
        relation_family = str(getattr(typed_spec, "relation_family", "") or "")
        operator_type = str(getattr(typed_spec, "operator_type", "") or "")
        answer_type = str(getattr(typed_spec, "answer_type", "") or "entity")
        granularity = str(getattr(typed_spec, "answer_granularity", "") or "default")

        facts = "; ".join(confirmed_values[:6])

        if operator_type == "boolean_conjunction":
            return (
                f"Question: {question}\n"
                f"Need evidence to verify yes/no.\n"
                f"Open target: {desc}\n"
                f"Known facts: {facts}\n"
                f"Return direct supporting evidence only."
            )

        if operator_type == "comparison":
            return (
                f"Question: {question}\n"
                f"Need the exact compared attribute.\n"
                f"Open target: {desc}\n"
                f"Known facts: {facts}\n"
                f"Answer type: {answer_type}"
            )

        if relation_family == "kinship":
            return (
                f"Question: {question}\n"
                f"Need missing kinship link.\n"
                f"Open target: {desc}\n"
                f"Known facts: {facts}\n"
                f"Answer type: person"
            )

        if relation_family == "location_admin":
            return (
                f"Question: {question}\n"
                f"Need exact location/admin evidence.\n"
                f"Open target: {desc}\n"
                f"Known facts: {facts}\n"
                f"Granularity: {granularity}\n"
                f"Answer type: location"
            )

        if relation_family == "entity_attribute":
            return (
                f"Question: {question}\n"
                f"Need missing attribute evidence.\n"
                f"Open target: {desc}\n"
                f"Known facts: {facts}\n"
                f"Answer type: {answer_type}"
            )

        return (
            f"Question: {question}\n"
            f"Need missing evidence for target.\n"
            f"Open target: {desc}\n"
            f"Known facts: {facts}\n"
            f"Answer type: {answer_type}"
        )
