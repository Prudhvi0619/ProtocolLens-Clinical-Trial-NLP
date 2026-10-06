"""Build TF-IDF and sentence-transformer comparable-trial indexes."""

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

from protocollens.retrieval import (  # noqa: E402
    DEFAULT_SENTENCE_MODEL,
    SemanticTrialRetriever,
    TfidfTrialRetriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparable-trial indexes")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "retrieval")
    parser.add_argument("--model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use an already downloaded sentence model without network checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise RuntimeError(f"No records found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tfidf = TfidfTrialRetriever(records)
    tfidf.save(args.output_dir / "tfidf.joblib")
    semantic = SemanticTrialRetriever(
        records,
        model_name=args.model,
        local_files_only=args.local_files_only,
    )
    semantic.save(args.output_dir / "semantic")
    metadata = {
        "records": len(records),
        "tfidf_vocabulary_size": len(tfidf.vectorizer.vocabulary_),
        "sentence_model": semantic.model_name,
        "embedding_dimensions": int(semantic.embeddings.shape[1]),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
