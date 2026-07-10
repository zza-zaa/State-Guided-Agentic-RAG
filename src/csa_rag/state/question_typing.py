from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from typing import List


@dataclass
class QuestionTypeSpec:
    relation_family: str
    operator_type: str
    answer_type: str
    answer_granularity: str
    chain_depth_hint: int
    typed_hints: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _norm(q: str) -> str:
    return " ".join(str(q or "").split()).strip().lower()


def infer_question_type(question: str) -> QuestionTypeSpec:
    q = _norm(question)

    relation_family = "other"
    operator_type = "lookup"
    answer_type = "entity"
    answer_granularity = "default"
    chain_depth_hint = 2
    typed_hints: List[str] = []

    # ------------------------------------------------------------
    # operator type
    # ------------------------------------------------------------
    if any(x in q for x in ["earlier", "later", "older", "younger", "died first", "born first"]):
        operator_type = "comparison"
        typed_hints.append("comparison over two candidates")
    elif "both " in q or q.startswith("are both") or q.startswith("do both"):
        operator_type = "boolean_conjunction"
        typed_hints.append("boolean conjunction over two sub-answers")
    elif any(x in q for x in ["father-in-law", "mother-in-law", "paternal", "maternal"]):
        operator_type = "kinship_composition"
        typed_hints.append("requires kinship composition")
    else:
        operator_type = "lookup"

    # ------------------------------------------------------------
    # relation family
    # ------------------------------------------------------------
    if any(x in q for x in ["father-in-law", "mother-in-law", "paternal", "maternal", "child of", "father of", "mother of", "spouse", "husband", "wife", "grandfather", "grandmother"]):
        relation_family = "kinship"
        answer_type = "person"
        typed_hints.append("kinship chain")

    elif any(x in q for x in ["award", "record label", "league", "position", "headquartered", "employer", "works at", "nationality", "country is from", "what country"]):
        relation_family = "entity_attribute"
        typed_hints.append("entity-to-attribute relation")

    elif any(x in q for x in ["county", "administrative territorial entity", "border with", "located", "where was", "place of birth", "place of death", "birthplace"]):
        relation_family = "location_admin"
        typed_hints.append("location/admin relation")

    elif operator_type in {"comparison", "boolean_conjunction"}:
        relation_family = "comparison_or_boolean"

    else:
        relation_family = "other"

    # ------------------------------------------------------------
    # answer type
    # ------------------------------------------------------------
    if q.startswith("who "):
        answer_type = "person"
    elif q.startswith("when ") or "date of" in q:
        answer_type = "date"
    elif q.startswith("where ") or "place of birth" in q or "place of death" in q:
        answer_type = "location"
    elif "what award" in q:
        answer_type = "award"
    elif "what league" in q:
        answer_type = "league"
    elif "what position" in q:
        answer_type = "position"
    elif "what nationality" in q or "which country" in q or "what country" in q:
        answer_type = "country_or_nationality"
    elif "what record label" in q:
        answer_type = "record_label"
    elif q.startswith("are ") or q.startswith("do "):
        answer_type = "boolean"

    # ------------------------------------------------------------
    # granularity
    # ------------------------------------------------------------
    if "county" in q:
        answer_granularity = "county_exact"
        typed_hints.append("prefer county-level answer")
    elif "administrative territorial entity" in q:
        answer_granularity = "admin_entity_exact"
        typed_hints.append("prefer administrative entity, not city")
    elif "place of birth" in q or "where was" in q and "born" in q:
        answer_granularity = "birthplace_locality"
        typed_hints.append("prefer birthplace locality")
    elif q.startswith("when ") or "date of" in q:
        answer_granularity = "full_date_preferred"
        typed_hints.append("prefer full date if available")
    elif answer_type == "person":
        answer_granularity = "minimal_entity_name"
        typed_hints.append("prefer minimal canonical person name")

    # ------------------------------------------------------------
    # depth hint
    # ------------------------------------------------------------
    if relation_family == "kinship":
        chain_depth_hint = 2 if any(x in q for x in ["father of", "mother of", "child of", "spouse"]) else 3
    elif operator_type in {"comparison", "boolean_conjunction"}:
        chain_depth_hint = 2
    elif relation_family in {"entity_attribute", "location_admin"}:
        chain_depth_hint = 2
    else:
        chain_depth_hint = 1

    return QuestionTypeSpec(
        relation_family=relation_family,
        operator_type=operator_type,
        answer_type=answer_type,
        answer_granularity=answer_granularity,
        chain_depth_hint=chain_depth_hint,
        typed_hints=typed_hints,
    )
