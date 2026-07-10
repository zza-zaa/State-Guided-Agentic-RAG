from __future__ import annotations

from typing import Any, List


class TypedSecondHopQueryBuilder:
    def build(
        self,
        question: str,
        slot: Any,
        dep_values: List[str],
        typed_spec: Any = None,
    ) -> str:
        desc = str(getattr(slot, "description", "") or "")
        parent = "; ".join(dep_values)
        relation_family = str(getattr(typed_spec, "relation_family", "") or "")
        operator_type = str(getattr(typed_spec, "operator_type", "") or "")
        answer_type = str(getattr(typed_spec, "answer_type", "") or "entity")
        granularity = str(getattr(typed_spec, "answer_granularity", "") or "default")

        # boolean / comparison
        if operator_type == "boolean_conjunction":
            return (
                f"Question: {question}\n"
                f"Need evidence for yes/no judgment.\n"
                f"Known entity or intermediate fact: {parent}\n"
                f"Target relation: {desc}\n"
                f"Return evidence that directly supports or refutes the predicate."
            )

        if operator_type == "comparison":
            return (
                f"Question: {question}\n"
                f"Need comparison attribute for: {parent}\n"
                f"Target relation: {desc}\n"
                f"Answer type: {answer_type}\n"
                f"Return the exact comparable attribute only."
            )

        # kinship
        if relation_family == "kinship":
            return (
                f"Question: {question}\n"
                f"Known family member/entity: {parent}\n"
                f"Target kinship relation: {desc}\n"
                f"Answer type: person\n"
                f"Return the exact related person only."
            )

        # location/admin
        if relation_family == "location_admin":
            return (
                f"Question: {question}\n"
                f"Known entity: {parent}\n"
                f"Target relation: {desc}\n"
                f"Answer type: location\n"
                f"Granularity: {granularity}\n"
                f"Return the exact location/admin unit only."
            )

        # entity attribute
        if relation_family == "entity_attribute":
            return (
                f"Question: {question}\n"
                f"Known entity: {parent}\n"
                f"Target relation: {desc}\n"
                f"Answer type: {answer_type}\n"
                f"Return only the target {answer_type}, not surrounding explanation."
            )

        # fallback
        return (
            f"Question: {question}\n"
            f"Known entity or intermediate fact: {parent}\n"
            f"Target relation: {desc}\n"
            f"Answer type: {answer_type}\n"
            f"Return the exact target evidence."
        )
