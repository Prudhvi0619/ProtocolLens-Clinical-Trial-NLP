"""Audit eligibility text for explicit registry-outcome language."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.modelling import (  # noqa: E402
    audit_outcome_language,
    sanitize_outcome_language,
)


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv"
    output_path = PROJECT_ROOT / "artifacts" / "models" / "text_leakage_audit.json"
    frame = pd.read_csv(input_path)
    raw = audit_outcome_language(frame["eligibility_criteria"])
    sanitized = audit_outcome_language(sanitize_outcome_language(frame["eligibility_criteria"]))
    by_status = {
        str(status): audit_outcome_language(group["eligibility_criteria"])
        for status, group in frame.groupby("overall_status")
    }
    result = {
        "field_audited": "eligibility_criteria",
        "interpretation": (
            "Raw exact-word matches can be legitimate protocol language. Every match is removed "
            "inside the NLP pipeline before TF-IDF fitting and inference; this does not prove "
            "prospective validity."
        ),
        "raw": raw,
        "sanitized_model_text": sanitized,
        "raw_by_status": by_status,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
