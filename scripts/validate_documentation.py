"""Verify that user-facing documentation matches generated evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
RETRIEVAL_DIR = PROJECT_ROOT / "artifacts" / "retrieval"


def normalized_document(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def require_phrases(path: Path, phrases: list[str]) -> None:
    document = normalized_document(path)
    missing = [phrase for phrase in phrases if phrase not in document]
    if missing:
        raise AssertionError(f"{path.name} has stale or missing claims: {missing}")


def main() -> None:
    summary = json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8"))
    best = summary["best_uncalibrated_metrics"]
    delta = summary["nlp_vs_structured_average_precision"]["logistic_regression"]
    leakage = json.loads((MODEL_DIR / "text_leakage_audit.json").read_text(encoding="utf-8"))
    proxy = json.loads((RETRIEVAL_DIR / "proxy_evaluation.json").read_text(encoding="utf-8"))
    relevance = json.loads((RETRIEVAL_DIR / "manual_evaluation.json").read_text(encoding="utf-8"))
    review = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation.csv")
    unique_pairs = int(review[["query_nct_id", "candidate_nct_id"]].drop_duplicates().shape[0])
    test_count = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests")).countTestCases()

    average_precision = best["average_precision"]
    roc_auc = best["roc_auc"]
    ap_lower = best["average_precision_ci_lower"]
    ap_upper = best["average_precision_ci_upper"]
    roc_lower = best["roc_auc_ci_lower"]
    roc_upper = best["roc_auc_ci_upper"]
    ap_delta = delta["mean_ap_difference"]
    delta_lower = delta["ci_95_lower"]
    delta_upper = delta["ci_95_upper"]
    raw_matches = leakage["raw"]["records_with_any_term"]
    sanitized_matches = leakage["sanitized_model_text"]["records_with_any_term"]
    semantic_proxy = proxy["metrics"]["semantic"]["precision_at_k"]
    tfidf_review = relevance["metrics"]["tfidf"]["precision_at_k"]
    semantic_review = relevance["metrics"]["semantic"]["precision_at_k"]

    require_phrases(
        PROJECT_ROOT / "README.md",
        [
            f"average precision: **{average_precision:.3f}**",
            f"95% CI {ap_lower:.3f}–{ap_upper:.3f}",
            f"ROC-AUC: **{roc_auc:.3f}**",
            f"95% CI {roc_lower:.3f}–{roc_upper:.3f}",
            f"**{ap_delta:+.3f}**",
            f"{unique_pairs} unique blinded comparisons",
            f"{test_count} tests",
            f"all {raw_matches} raw exact outcome-word matches",
            f"TF-IDF Precision@5 was **{tfidf_review:.3f}**",
            f"**{semantic_review:.3f}** for semantic retrieval",
            "not an independent human gold standard",
        ],
    )
    require_phrases(
        PROJECT_ROOT / "docs" / "MODEL_CARD.md",
        [
            f"Average precision: {average_precision:.3f}",
            f"ROC-AUC: {roc_auc:.3f}",
            f"{ap_delta:+.3f} (paired bootstrap 95% CI {delta_lower:+.3f} to {delta_upper:+.3f})",
            f"raw cohort contains {raw_matches} matches",
            "post-sanitization audit contains zero matches"
            if sanitized_matches == 0
            else f"post-sanitization audit contains {sanitized_matches} matches",
            f"Precision@5 {tfidf_review:.3f} for TF-IDF",
            f"{semantic_review:.3f} for semantic retrieval",
            "not an independent human gold standard",
        ],
    )
    require_phrases(
        PROJECT_ROOT / "docs" / "RESUME_AND_INTERVIEW.md",
        [
            f"semantic retrieval reached {semantic_proxy:.3f}",
            f"eligibility NLP achieved {average_precision:.3f}",
            f"{ap_delta:+.3f} paired-bootstrap improvement",
            f"95% CI {delta_lower:+.3f} to {delta_upper:+.3f}",
            f"All {unique_pairs} unique AI-assisted comparisons",
            f"Precision@5 {tfidf_review:.3f}",
            f"semantic retrieval ({semantic_review:.3f})",
        ],
    )
    require_phrases(
        PROJECT_ROOT / "docs" / "COMPLETION_AUDIT.md",
        [
            f"{unique_pairs} unique blinded comparisons",
            f"{test_count} passing tests",
            f"{raw_matches} raw exact-word matches removed",
        ],
    )
    require_phrases(
        PROJECT_ROOT / "docs" / "EVALUATION_GUIDE.md",
        [f"{unique_pairs} unique blinded comparisons"],
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "average_precision": round(average_precision, 3),
                "roc_auc": round(roc_auc, 3),
                "nlp_ap_delta": round(ap_delta, 3),
                "unique_review_pairs": unique_pairs,
                "tests_documented": test_count,
                "raw_outcome_language_matches": raw_matches,
                "sanitized_model_text_matches": sanitized_matches,
                "tfidf_ai_assisted_precision_at_5": tfidf_review,
                "semantic_ai_assisted_precision_at_5": semantic_review,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
