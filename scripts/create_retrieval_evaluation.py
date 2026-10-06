"""Create a blinded relevance worksheet for both retrieval methods."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.retrieval import (  # noqa: E402
    SemanticTrialRetriever,
    TfidfTrialRetriever,
    select_review_query_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create retrieval review sheet")
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "retrieval"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation.csv",
    )
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_labels: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    if args.output.exists():
        with args.output.open("r", newline="", encoding="utf-8-sig") as handle:
            for existing in csv.DictReader(handle):
                key = (
                    existing.get("query_nct_id", ""),
                    existing.get("method", ""),
                    existing.get("candidate_nct_id", ""),
                )
                existing_labels[key] = (
                    existing.get("relevance_0_2", ""),
                    existing.get("reviewer_note", ""),
                    existing.get("review_source", ""),
                )
    tfidf = TfidfTrialRetriever.load(args.artifact_dir / "tfidf.joblib")
    semantic = SemanticTrialRetriever.load(args.artifact_dir / "semantic")
    records_by_id = {record["nct_id"]: record for record in tfidf.records}
    rows: list[dict] = []
    for query_id in select_review_query_ids(tfidf.records):
        query = records_by_id[query_id]
        for method, retriever in (("tfidf", tfidf), ("semantic", semantic)):
            for candidate in retriever.search_by_id(query_id, k=args.k):
                candidate_record = records_by_id[candidate["nct_id"]]
                prior_label, prior_note, prior_source = existing_labels.get(
                    (query_id, method, candidate["nct_id"]), ("", "", "")
                )
                rows.append(
                    {
                        "query_nct_id": query_id,
                        "query_area": query.get("query_conditions"),
                        "query_title": query.get("brief_title"),
                        "query_phase": query.get("phases"),
                        "query_intervention_types": query.get("intervention_types"),
                        "query_eligibility": query.get("eligibility_criteria"),
                        "method": method,
                        "rank": candidate["rank"],
                        "candidate_nct_id": candidate["nct_id"],
                        "candidate_area": candidate["query_conditions"],
                        "candidate_title": candidate["brief_title"],
                        "candidate_phase": candidate_record.get("phases"),
                        "candidate_intervention_types": candidate_record.get("intervention_types"),
                        "candidate_eligibility": candidate_record.get("eligibility_criteria"),
                        "candidate_status": candidate["overall_status"],
                        "similarity": candidate["similarity"],
                        "relevance_0_2": prior_label,
                        "reviewer_note": prior_note,
                        "review_source": prior_source,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preserved = sum(row["relevance_0_2"] not in ("", None) for row in rows)
    print(f"Created {len(rows)} judgments at {args.output}")
    print(f"Preserved {preserved} existing judgments")
    print("Label relevance_0_2 as 0=irrelevant, 1=partly relevant, 2=highly relevant")


if __name__ == "__main__":
    main()
