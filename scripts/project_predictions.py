from __future__ import annotations

import json
import re
from pathlib import Path
import typer

app = typer.Typer()

MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
COUNTRY_TAILS = {
    "japan", "china", "people's republic of china", "united states", "usa", "u.s.", "canada",
    "france", "italy", "germany", "egypt", "england", "uk", "united kingdom", "serbia"
}


def normalize_ws(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def get_answer_target_value(state: dict) -> str | None:
    if not isinstance(state, dict):
        return None
    for s in state.get("slots", []) or []:
        if s.get("target_role") == "answer_target":
            v = s.get("value")
            if v not in [None, ""]:
                return str(v)
    return None


def get_evidence_texts(row: dict) -> list[str]:
    out = []
    for ev in row.get("retrieved_evidence_top5", []) or []:
        if isinstance(ev, dict):
            t = ev.get("text")
            if t:
                out.append(str(t))
    return out


def strip_trailing_country(loc: str) -> str:
    parts = [p.strip() for p in loc.split(",")]
    if len(parts) >= 2:
        last = parts[-1].lower()
        if last in COUNTRY_TAILS:
            parts = parts[:-1]
    return ", ".join(parts)


def county_only(loc: str) -> str:
    m = re.search(r"([A-Z][A-Za-z .'-]*County)", loc)
    if m:
        return normalize_ws(m.group(1))
    return normalize_ws(loc)


def project_when(question: str, base: str, evidence_texts: list[str]) -> str:
    q = question.lower()
    if not ("when" in q or "ended" in q or "date" in q or "died" in q or "born" in q):
        return base

    base_year = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", base)
    if not base_year:
        return base
    year = base_year.group(1)

    # special: war/question-ended range like "... November 1917 – October 1922 ..."
    for t in evidence_texts:
        m = re.search(rf"[–-]\s*({MONTHS}\s+{year})", t)
        if m:
            return normalize_ws(m.group(1))

    # general month-year / full date containing the same year
    for t in evidence_texts:
        m = re.search(rf"\b({MONTHS}\s+\d{{1,2}},\s*{year}|{MONTHS}\s+{year})\b", t)
        if m:
            return normalize_ws(m.group(1))

    return base


def extract_after_pattern(texts: list[str], pattern: str) -> str | None:
    for t in texts:
        m = re.search(pattern, t, flags=re.I)
        if m:
            return normalize_ws(m.group(1))
    return None


def project_location(question: str, base: str, evidence_texts: list[str]) -> str:
    q = question.lower()
    base = normalize_ws(base)

    # county questions
    if "county" in q:
        return county_only(base)

    # birthplace / born / place of birth -> smaller span
    if "place of birth" in q or "where was" in q and "born" in q or "birthplace" in q:
        span = extract_after_pattern(
            evidence_texts,
            rf"born in ([A-Z][^.;]*?)(?:,? as\b|\.|;)"
        )
        if span:
            span = strip_trailing_country(span)
            # conservative: for explicit place-of-birth questions, keep first segment
            first = span.split(",")[0].strip()
            if first:
                return normalize_ws(first)
        base = strip_trailing_country(base)
        first = base.split(",")[0].strip()
        return normalize_ws(first) if first else base

    # hail from / formed in / origin -> keep city+region, drop country
    if "hail from" in q or "origin" in q or "formed in" in q:
        span = extract_after_pattern(
            evidence_texts,
            rf"(?:formed in|from) ([A-Z][^.;]*?)(?: in \d{{4}}|\.|;)"
        )
        if span:
            return normalize_ws(strip_trailing_country(span))
        return normalize_ws(strip_trailing_country(base))

    # educated where -> keep shortest school-like span
    if "educated" in q or "school" in q:
        if " in " in base:
            base = base.split(" in ")[0].strip()
        return normalize_ws(base)

    return base


def project_generic(base: str) -> str:
    base = normalize_ws(base)
    # remove leading explanatory sentence wrappers
    for prefix in [
        "The answer is ",
        "The confirmed answer is ",
        "It is ",
    ]:
        if base.startswith(prefix):
            base = base[len(prefix):].strip()

    # strip terminal punctuation
    return base.rstrip(" .;")


def project_answer(row: dict) -> str:
    question = normalize_ws(row.get("question", ""))
    state = row.get("state", {})
    base = (
        get_answer_target_value(state)
        or row.get("pred")
        or row.get("raw_pred")
        or ""
    )
    base = normalize_ws(base)
    evidence_texts = get_evidence_texts(row)

    if not base:
        return base

    projected = base
    projected = project_when(question, projected, evidence_texts)
    projected = project_location(question, projected, evidence_texts)
    projected = project_generic(projected)
    return projected


@app.command()
def main(
    input_path: Path = typer.Option(..., exists=True),
    output_path: Path = typer.Option(...),
):
    changed = 0
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f, output_path.open("w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            old_pred = row.get("pred")
            new_pred = project_answer(row)

            if new_pred != old_pred:
                changed += 1

            row["pred_projected"] = new_pred
            row["pred"] = new_pred
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] wrote {total} rows to {output_path}")
    print(f"[done] changed predictions: {changed}")


if __name__ == "__main__":
    app()
