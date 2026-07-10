from __future__ import annotations

import re


def infer_expected_answer_type(question: str, slot_description: str = "") -> str:
    q = f"{question} {slot_description}".lower().strip()

    if q.startswith("who ") or any(x in q for x in [" spouse", " father", " mother", " child", " sibling", " brother", " sister"]):
        return "person"
    if q.startswith("when ") or " what year" in q or "founded" in q or "birth date" in q:
        return "date"
    if q.startswith("where "):
        return "location"
    if "which county" in q or "in which county" in q:
        return "county"
    if "what province" in q or "which province" in q:
        return "province"
    if "record label" in q or "signed to" in q:
        return "record_label"
    if "award" in q or "prize" in q or "medal" in q:
        return "award"
    if "team" in q:
        return "team"
    if "instrument" in q:
        return "instrument"
    if "era" in q:
        return "era"
    if q.startswith("is ") or q.startswith("are ") or q.startswith("did ") or q.startswith("does "):
        return "boolean"

    return "other"


def infer_cardinality(question: str, slot_description: str = "") -> str:
    q = f"{question} {slot_description}".lower().strip()
    multi_markers = [
        "which teams",
        "what teams",
        "which children",
        "what children",
        "which siblings",
        "what siblings",
        "list",
        "names of",
    ]
    if any(m in q for m in multi_markers):
        return "multi"
    return "single"


def _looks_like_date(value: str) -> bool:
    value = value.strip()
    if re.search(r"\b\d{4}\b", value):
        return True
    if re.search(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", value):
        return True
    if re.search(r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b", value):
        return True
    return False


def _looks_multi_valued(value: str) -> bool:
    v = value.strip()
    if ";" in v:
        return True
    if " and " in v:
        return True
    if "," in v and len(v.split(",")) >= 2:
        return True
    return False


def is_value_type_compatible(question: str, value: str | None, expected_type: str = "other") -> bool:
    if value is None:
        return False

    v = str(value).strip()
    if v == "":
        return False

    low = v.lower()

    if low in {"unknown", "none", "cannot be determined"}:
        return False

    if expected_type == "boolean":
        return low in {"yes", "no", "true", "false"}

    if expected_type == "date":
        return _looks_like_date(v)

    if expected_type == "county":
        return "county" in low

    if expected_type == "province":
        return "province" in low or "state" in low or "governorate" in low

    if expected_type == "record_label":
        return any(x in low for x in ["records", "recordings", "label", "entertainment"])

    if expected_type == "award":
        return any(x in low for x in ["award", "prize", "medal", "oscar", "academy"])

    if expected_type == "instrument":
        return not _looks_multi_valued(v)

    if expected_type == "person":
        if low in {"yes", "no", "true", "false"}:
            return False
        return True

    return True


def is_answer_target_value_valid(
    question: str,
    value: str | None,
    expected_type: str = "other",
    cardinality: str = "single",
) -> bool:
    if value is None:
        return False
    v = str(value).strip()
    if v == "":
        return False
    if not is_value_type_compatible(question, v, expected_type):
        return False
    if cardinality == "single" and _looks_multi_valued(v):
        return False
    return True


# 兼容旧版本 calibration.py 的调用方式
def is_invalid_answer_for_question(question: str, value: str | None) -> bool:
    expected_type = infer_expected_answer_type(question)
    cardinality = infer_cardinality(question)
    return not is_answer_target_value_valid(
        question=question,
        value=value,
        expected_type=expected_type,
        cardinality=cardinality,
    )
