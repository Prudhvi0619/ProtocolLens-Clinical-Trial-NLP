"""Post-outcome descriptive summaries that are never used as model features."""

from __future__ import annotations

import re

REASON_PATTERNS = [
    ("Recruitment or enrollment", re.compile(r"\b(?:recruit|enrol|accrual)\w*\b", re.I)),
    ("Safety or tolerability", re.compile(r"\b(?:safety|adverse|toxicit|tolerab)\w*\b", re.I)),
    (
        "Efficacy or futility",
        re.compile(r"\b(?:efficacy|ineffective|futil|lack of effect|benefit)\w*\b", re.I),
    ),
    (
        "Funding or business",
        re.compile(r"\b(?:fund|financ|business|commercial|budget|resource)\w*\b", re.I),
    ),
    (
        "External or operational",
        re.compile(r"\b(?:covid|pandemic|supply|operational|regulatory|logistic)\w*\b", re.I),
    ),
    (
        "Sponsor or strategic decision",
        re.compile(r"\b(?:sponsor|strategic|company|investigator decision)\w*\b", re.I),
    ),
]


def categorize_termination_reason(value: str | None) -> str:
    """Map free-text stop reasons to transparent descriptive categories."""
    if value is None or not str(value).strip():
        return "Not reported"
    text = str(value).strip()
    for category, pattern in REASON_PATTERNS:
        if pattern.search(text):
            return category
    return "Other stated reason"
