"""Run a weak-label retrieval sanity check based on therapeutic-area overlap."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.retrieval import (  # noqa: E402
    SemanticTrialRetriever,
    TfidfTrialRetriever,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


def area_set(value: str | None) -> set[str]:
    return {part.strip() for part in str(value or "").split("|") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Weak-label retrieval sanity check")
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "retrieval"
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    tfidf = TfidfTrialRetriever.load(args.artifact_dir / "tfidf.joblib")
    semantic = SemanticTrialRetriever.load(args.artifact_dir / "semantic")
    rng = np.random.default_rng(42)
    sample_size = min(args.sample_size, len(tfidf.records))
    sample_indices = rng.choice(len(tfidf.records), size=sample_size, replace=False)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for method, retriever in (("tfidf", tfidf), ("semantic", semantic)):
        for index in sample_indices:
            query = retriever.records[int(index)]
            query_areas = area_set(query.get("query_conditions"))
            results = retriever.search_by_id(query["nct_id"], k=args.k)
            relevance = [
                int(bool(query_areas & area_set(result.get("query_conditions"))))
                for result in results
            ]
            metrics[method]["precision_at_k"].append(precision_at_k(relevance, args.k))
            metrics[method]["mrr"].append(reciprocal_rank(relevance))
            metrics[method]["ndcg_at_k"].append(ndcg_at_k(relevance, args.k))
    summary = {
        "warning": (
            "Therapeutic-area overlap is a weak sanity-check label, not human "
            "relevance judgment. Use retrieval_evaluation.csv for manual review."
        ),
        "sample_size": sample_size,
        "k": args.k,
        "metrics": {
            method: {
                metric: round(float(np.mean(values)), 4)
                for metric, values in method_metrics.items()
            }
            for method, method_metrics in metrics.items()
        },
    }
    output = args.artifact_dir / "proxy_evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
