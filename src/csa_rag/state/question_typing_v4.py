from __future__ import annotations

from dataclasses import dataclass, asdict
import json
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


def infer_question_type_v4(question: str) -> QuestionTypeSpec:
    q = _norm(question)

    relation_family = "other"
    operator_type = "lookup"
    answer_type = "entity"
    answer_granularity = "default"
    chain_depth_hint = 1
    typed_hints: List[str] = []

    comparison_cues = [
        "came out first", "came out earlier", "released earlier", "released later",
        "died first", "died earlier", "died later",
        "born first", "born earlier", "born later",
        "younger", "older", "higher ", "lower ",
        "which film has the director", "who died first",
        "which writer was from ", "which performance act has "
    ]
    boolean_cues = [
        "are both", "do both", "both ",
        "same country", "same nationality", "same conference", "same neighborhood"
    ]
    kinship_comp_cues = [
        "father-in-law", "mother-in-law",
        "paternal grandfather", "paternal grandmother",
        "maternal grandfather", "maternal grandmother"
    ]

    if any(cue in q for cue in kinship_comp_cues):
        operator_type = "kinship_composition"
        typed_hints.append("requires kinship composition")
    elif any(cue in q for cue in boolean_cues):
        operator_type = "boolean_conjunction"
        typed_hints.append("boolean conjunction over sub-answers")
    elif any(cue in q for cue in comparison_cues) or (" or " in q and q.startswith(("which ", "who "))):
        operator_type = "comparison"
        typed_hints.append("comparison over candidates")

    kinship_terms = [
        "father-in-law", "mother-in-law",
        "paternal grandfather", "paternal grandmother",
        "maternal grandfather", "maternal grandmother",
        "grandfather", "grandmother",
        "spouse", "husband", "wife",
        "mother of", "father of", "child of"
    ]
    false_kinship_phrases = [
        "father of modern", "mother of invention"
    ]

    if any(t in q for t in kinship_terms) and not any(bad in q for bad in false_kinship_phrases):
        relation_family = "kinship"
        typed_hints.append("kinship chain")
    elif operator_type in {"comparison", "boolean_conjunction"}:
        relation_family = "comparison_or_boolean"
    elif any(t in q for t in [
        "award", "record label", "league", "position", "headquartered", "employer", "works at",
        "founded", "founded by", "founder", "owns", "owner", "manufacturer", "distributed",
        "goal of", "notable work", "instrument", "what company", "which company",
        "conference", "formerly known as", "served during what years", "voted to be",
        "voiced what", "control the program", "represented", "founded in what year",
        "cause of death", "nationality", "what nationality", "which country", "what country",
        "how many people", "seat how many", "population of how many inhabitants",
        "government position", "ice hockey position"
    ]):
        relation_family = "entity_attribute"
        typed_hints.append("entity-to-attribute relation")
    elif any(t in q for t in [
        "where ", "place of birth", "place of death", "birthplace",
        "county", "borough", "province", "administrative territorial entity",
        "share a border", "border with", "located in", "located?", "same neighborhood"
    ]):
        relation_family = "location_admin"
        typed_hints.append("location/admin relation")

    if q.startswith("who "):
        answer_type = "person"
    elif q.startswith("when ") or "date of" in q or "what year" in q or "during what years" in q:
        answer_type = "date"
    elif "how many" in q or "population of how many" in q or "can seat how many" in q:
        answer_type = "number"
    elif q.startswith("where ") or "in which county" in q or "in which borough" in q or "place of birth" in q or "place of death" in q:
        answer_type = "location"
    elif "what award" in q:
        answer_type = "award"
    elif "what league" in q:
        answer_type = "league"
    elif "what record label" in q:
        answer_type = "record_label"
    elif "what government position" in q or "which professional ice hockey position" in q or "what position" in q:
        answer_type = "position"
    elif "cause of death" in q or ("why did" in q and "die" in q):
        answer_type = "cause_of_death"
    elif "instrument" in q:
        answer_type = "instrument"
    elif "what company" in q or "which company" in q:
        answer_type = "organization"
    elif "which country" in q or "what nationality" in q or "what country" in q or "same nationality" in q:
        answer_type = "country_or_nationality"
    elif operator_type == "boolean_conjunction":
        answer_type = "boolean"
    elif q.startswith("are ") or q.startswith("do "):
        answer_type = "boolean"
    elif operator_type == "comparison" and q.startswith("who "):
        answer_type = "person"
    elif operator_type == "comparison":
        answer_type = "entity"

    # refine answer type for some entity_attribute patterns
    if relation_family == "entity_attribute":
        if "founded by" in q or "who founded" in q or "founder" in q:
            answer_type = "person"
        elif "position" in q:
            answer_type = "position"

    if "county" in q:
        answer_granularity = "county_exact"
        typed_hints.append("prefer county-level answer")
    elif "borough" in q:
        answer_granularity = "borough_exact"
        typed_hints.append("prefer borough-level answer")
    elif "administrative territorial entity" in q:
        answer_granularity = "admin_entity_exact"
        typed_hints.append("prefer administrative entity, not city")
    elif "place of birth" in q or ("where was" in q and "born" in q):
        answer_granularity = "birthplace_locality"
        typed_hints.append("prefer birthplace locality")
    elif q.startswith("when ") or "date of" in q or "what year" in q:
        answer_granularity = "full_date_preferred"
        typed_hints.append("prefer full date if available")
    elif answer_type == "person":
        answer_granularity = "minimal_entity_name"
        typed_hints.append("prefer minimal canonical person name")

    nested_cues = [
        "director of", "performer of", "author of", "composer of",
        "owner of", "manufacturer of", "team that plays in",
        "company that distributed", "person who is part of",
        "father of", "mother of", "child of", "spouse of"
    ]

    if operator_type in {"comparison", "boolean_conjunction"}:
        chain_depth_hint = 2
    elif operator_type == "kinship_composition":
        chain_depth_hint = 3
    elif relation_family == "kinship":
        chain_depth_hint = 2
    elif any(x in q for x in nested_cues):
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
