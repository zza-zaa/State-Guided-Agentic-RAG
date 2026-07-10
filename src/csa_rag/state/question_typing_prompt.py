from __future__ import annotations

from typing import Any


def format_typed_spec(spec: Any) -> str:
    return (
        "[TypedQuestionSpec]\n"
        f"relation_family: {spec.relation_family}\n"
        f"operator_type: {spec.operator_type}\n"
        f"answer_type: {spec.answer_type}\n"
        f"answer_granularity: {spec.answer_granularity}\n"
        f"chain_depth_hint: {spec.chain_depth_hint}\n"
        f"typed_hints: {', '.join(spec.typed_hints)}\n"
        "[/TypedQuestionSpec]"
    )
