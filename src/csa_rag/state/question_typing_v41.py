from __future__ import annotations

from csa_rag.state.question_typing_v4 import infer_question_type_v4 as _infer_v4


def infer_question_type_v41(question: str):
    spec = _infer_v4(question)

    # tiny fix:
    # boolean-conjunction questions should always have boolean answer type,
    # even if a subtype cue like "nationality" appears.
    if spec.operator_type == "boolean_conjunction":
        spec.answer_type = "boolean"

    return spec
