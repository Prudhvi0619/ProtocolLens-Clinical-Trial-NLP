"""Score a completed retrieval relevance evaluation sheet."""

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

from protocollens.retrieval import summarize_judgments  # noqa: E402


def evaluation_provenance(rows: list[dict[str, str]]) -> dict[str, object]:
    """Describe whether completed labels are human or AI assisted."""
    sources = {row.get("review_source", "").strip() for row in rows}
    sources.discard("")
    if sources == {"ai_assisted_llm"}:
        return {
            "review_type": "AI-assisted relevance judgments",
            "review_source": "ai_assisted_llm",
            "independent_human_review": False,
            "limitation": (
                "These judgments are an LLM-based qualitative assessment and are not "
                "an independent human gold standard."
            ),
        }
    return {
        "review_type": "manual human relevance judgments",
        "review_source": "manual_human",
        "independent_human_review": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval relevance labels")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation.csv",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "retrieval" / "manual_evaluation.json",
    )
    args = parser.parse_args()
    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row for row in rows if row["relevance_0_2"].strip() == ""]
    if missing:
        raise RuntimeError(f"Complete all labels first; {len(missing)} are blank")

    summary = summarize_judgments(rows, args.k)
    result = {
        **evaluation_provenance(rows),
        "judgments": len(rows),
        "k": args.k,
        "metrics": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved retrieval evaluation: {args.output}")


if __name__ == "__main__":
    main()
