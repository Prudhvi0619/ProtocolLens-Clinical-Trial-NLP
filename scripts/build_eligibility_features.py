"""Parse eligibility text and append transparent complexity features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.eligibility import criteria_as_dicts, eligibility_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build eligibility NLP features")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "trials.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv",
    )
    parser.add_argument(
        "--criteria-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "parsed_criteria.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input}")

    enriched_rows: list[dict] = []
    parsed_records: list[dict] = []
    for row in rows:
        criteria_text = row.get("eligibility_criteria")
        enriched_rows.append({**row, **eligibility_features(criteria_text)})
        parsed_records.append(
            {
                "nct_id": row.get("nct_id"),
                "criteria": criteria_as_dicts(criteria_text),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched_rows[0]))
        writer.writeheader()
        writer.writerows(enriched_rows)
    with args.criteria_output.open("w", encoding="utf-8") as handle:
        for record in parsed_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_complexity = sum(float(row["complexity_index"]) for row in enriched_rows) / len(
        enriched_rows
    )
    print(f"Processed {len(enriched_rows)} studies")
    print(f"Average descriptive complexity index: {avg_complexity:.2f}")
    print(f"Feature table: {args.output}")


if __name__ == "__main__":
    main()
