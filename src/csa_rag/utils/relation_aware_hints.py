from __future__ import annotations

import re


def _clean(q: str) -> str:
    return " ".join(str(q).strip().split())


def infer_question_family(question: str) -> str:
    q = _clean(question).lower()

    if any(x in q for x in [
        "paternal grandmother", "paternal grandfather",
        "maternal grandmother", "maternal grandfather",
        "'s father", "'s mother", "'s spouse", "'s child", "'s sibling"
    ]):
        return "kinship_chain"

    if any(x in q for x in [
        "population of", "in which county", "which county",
        "which province", "what province", "administrative territorial entity",
        "share a border with", "headquartered", "employer", "country has a population"
    ]):
        return "administrative_chain"

    if q.startswith("where are ") and " and " in q:
        return "common_super_location"

    if any(x in q for x in [
        "which film", "which song", "which album", "older", "younger",
        "born later", "born first", "came out first", "released earlier",
        "released later", "died earlier", "died later"
    ]):
        return "comparison"

    if any(x in q for x in [
        "director of", "screenwriter of", "performer of", "author of",
        "star of", "the spouse of", "the child of", "the wife of", "the husband of"
    ]):
        return "entity_attribute_chain"

    return "generic_multi_hop"


def build_relation_aware_hint(question: str) -> str:
    family = infer_question_family(question)

    hints = [f"question_family={family}"]

    if family == "kinship_chain":
        hints.extend([
            "Expand kinship composition explicitly as a chain of binary relations.",
            "Examples:",
            "- paternal grandmother = father's mother",
            "- paternal grandfather = father's father",
            "- maternal grandmother = mother's mother",
            "- maternal grandfather = mother's father",
            "Do not collapse the chain into a single ambiguous slot.",
            "Use one bridge slot per hop and make the final kinship hop the answer_target."
        ])

    elif family == "administrative_chain":
        hints.extend([
            "Preserve the administrative/geographic chain explicitly.",
            "Examples:",
            "- place -> country -> population",
            "- birthplace -> county",
            "- city -> state/province -> bordering entity",
            "- employer -> headquarters",
            "Do not stop after resolving the first bridge entity."
        ])

    elif family == "common_super_location":
        hints.extend([
            "Resolve both entities independently first, then derive the shared higher-level location.",
            "Use two dependent location slots, then one answer_target slot for the common/superordinate place.",
            "Do not return the two raw locations as the final answer if the question asks where both are located."
        ])

    elif family == "comparison":
        hints.extend([
            "Preserve the original compared candidates explicitly.",
            "Use separate slots for candidate A and candidate B.",
            "Use dependent slots for the compared attribute of each candidate.",
            "Make the final answer_target represent the selected candidate, not the compared attribute value."
        ])

    elif family == "entity_attribute_chain":
        hints.extend([
            "First resolve the referenced entity exactly, then ask the target attribute about that resolved entity.",
            "Keep work attributes separate from person attributes.",
            "Examples:",
            "- director of film -> birthplace of that director",
            "- performer of song -> record label of performer, not release label of the work",
            "- star of film -> spouse of that star"
        ])

    else:
        hints.extend([
            "Prefer explicit bridge-to-answer chains.",
            "If the question is compositional, expose intermediate entities rather than implicit reasoning."
        ])

    return "\n".join(hints)
