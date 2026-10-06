"""Rule-based eligibility parsing and transparent complexity features."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean

SECTION_PATTERN = re.compile(
    r"^(?:key\s+)?(?P<section>inclusion|exclusion)\s+criteria\s*:?(?P<rest>.*)$",
    re.IGNORECASE,
)
BULLET_PATTERN = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s*")
WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
NUMBER_PATTERN = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?)(?!\w)")
COMPARATOR_PATTERN = re.compile(
    r"(?:<=|>=|<|>|≤|≥|less than|greater than|at least|at most|up to)",
    re.IGNORECASE,
)
TEMPORAL_PATTERN = re.compile(r"\b(?:hour|day|week|month|year)s?\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|absence|excluded?|ineligible|unable|must\s+not)\b",
    re.IGNORECASE,
)
CLINICAL_MEASURE_PATTERN = re.compile(
    r"\b(?:hemoglobin|platelets?|creatinine|bilirubin|albumin|ast|alt|egfr|"
    r"blood pressure|bmi|ecog|hba1c|glucose|oxygen saturation)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Criterion:
    section: str
    text: str


def _clean_line(line: str) -> str:
    line = BULLET_PATTERN.sub("", line.strip())
    return re.sub(r"\s+", " ", line).strip(" ;-\t")


def parse_eligibility(text: str | None) -> list[Criterion]:
    """Split eligibility text into inclusion, exclusion, or unknown criteria.

    This is deliberately interpretable. It detects section headers and treats
    each non-empty bullet/line as a criterion rather than inventing clinical
    entities through an opaque model.
    """
    if not text or not text.strip():
        return []

    current_section = "unknown"
    criteria: list[Criterion] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        heading_candidate = BULLET_PATTERN.sub("", stripped).strip()
        heading = SECTION_PATTERN.match(heading_candidate)
        if heading:
            current_section = heading.group("section").lower()
            remainder = _clean_line(heading.group("rest"))
            if remainder:
                criteria.append(Criterion(current_section, remainder))
            continue
        cleaned = _clean_line(stripped)
        if cleaned:
            criteria.append(Criterion(current_section, cleaned))
    return criteria


def eligibility_features(text: str | None) -> dict[str, int | float]:
    """Create auditable counts plus a documented 0–100 complexity index."""
    safe_text = text or ""
    criteria = parse_eligibility(safe_text)
    word_count = len(WORD_PATTERN.findall(safe_text))
    inclusion_count = sum(item.section == "inclusion" for item in criteria)
    exclusion_count = sum(item.section == "exclusion" for item in criteria)
    known_count = inclusion_count + exclusion_count
    criterion_lengths = [len(WORD_PATTERN.findall(item.text)) for item in criteria]
    numeric_count = len(NUMBER_PATTERN.findall(safe_text))
    comparator_count = len(COMPARATOR_PATTERN.findall(safe_text))
    temporal_count = len(TEMPORAL_PATTERN.findall(safe_text))
    negation_count = len(NEGATION_PATTERN.findall(safe_text))
    clinical_measure_count = len(CLINICAL_MEASURE_PATTERN.findall(safe_text))
    exclusion_ratio = exclusion_count / known_count if known_count else 0.0

    # A descriptive index, not a learned risk score. Each component is capped so
    # one unusually long protocol cannot dominate the profile.
    complexity_index = 100 * (
        0.25 * min(len(criteria) / 25, 1)
        + 0.20 * min(word_count / 600, 1)
        + 0.15 * min(numeric_count / 15, 1)
        + 0.15 * min(temporal_count / 12, 1)
        + 0.10 * min(comparator_count / 10, 1)
        + 0.10 * min(clinical_measure_count / 10, 1)
        + 0.05 * exclusion_ratio
    )

    return {
        "eligibility_word_count": word_count,
        "eligibility_character_count": len(safe_text),
        "criterion_count": len(criteria),
        "inclusion_count": inclusion_count,
        "exclusion_count": exclusion_count,
        "unclassified_count": len(criteria) - known_count,
        "average_criterion_words": round(mean(criterion_lengths), 3) if criterion_lengths else 0.0,
        "exclusion_ratio": round(exclusion_ratio, 4),
        "numeric_constraint_count": numeric_count,
        "comparator_count": comparator_count,
        "temporal_constraint_count": temporal_count,
        "negation_count": negation_count,
        "clinical_measure_count": clinical_measure_count,
        "complexity_index": round(complexity_index, 2),
    }


def criteria_as_dicts(text: str | None) -> list[dict[str, str]]:
    return [asdict(item) for item in parse_eligibility(text)]
