"""Download and normalize ClinicalTrials.gov studies."""

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

from protocollens.clinicaltrials import (  # noqa: E402
    ClinicalTrialsClient,
    FetchConfig,
    normalize_study,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a completed-versus-terminated clinical-trial dataset."
    )
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Disease or condition; repeat this option for multiple areas",
    )
    parser.add_argument(
        "--per-status",
        type=int,
        default=100,
        help="Maximum studies for each condition/status pair",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.output_dir / "raw"
    processed_dir = args.output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    client = ClinicalTrialsClient()
    studies_by_id: dict[str, dict] = {}
    matched_conditions: dict[str, set[str]] = {}
    query_counts: dict[str, int] = {}
    for condition in args.condition:
        for requested_status in ("COMPLETED", "TERMINATED"):
            config = FetchConfig(
                condition=condition,
                max_studies=args.per_status,
                page_size=args.page_size,
                statuses=(requested_status,),
            )
            fetched = list(client.iter_studies(config))
            query_counts[f"{condition}::{requested_status}"] = len(fetched)
            for study in fetched:
                nct_id = (
                    study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                )
                if not nct_id:
                    continue
                studies_by_id.setdefault(nct_id, study)
                matched_conditions.setdefault(nct_id, set()).add(condition)

    studies = list(studies_by_id.values())
    if not studies:
        raise RuntimeError("No studies matched the query")
    rows = []
    for study in studies:
        row = normalize_study(study)
        row["query_conditions"] = "|".join(sorted(matched_conditions[row["nct_id"]]))
        rows.append(row)

    raw_path = raw_dir / "studies.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for study in studies:
            handle.write(json.dumps(study, ensure_ascii=False) + "\n")

    csv_path = processed_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source": "ClinicalTrials.gov API v2",
        "fetched_at_utc": utc_timestamp(),
        "conditions": args.condition,
        "requested_statuses": ["COMPLETED", "TERMINATED"],
        "requested_per_condition_status": args.per_status,
        "query_counts_before_deduplication": query_counts,
        "records_received": len(rows),
        "class_counts": {
            status: sum(row["overall_status"] == status for row in rows)
            for status in ("COMPLETED", "TERMINATED")
        },
        "raw_file": str(raw_path.resolve()),
        "normalized_file": str(csv_path.resolve()),
    }
    metadata_path = raw_dir / "fetch_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {len(rows)} studies")
    print(f"Class counts: {metadata['class_counts']}")
    print(f"Normalized data: {csv_path}")


if __name__ == "__main__":
    main()
