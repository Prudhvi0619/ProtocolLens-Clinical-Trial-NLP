"""Fast acceptance checks for generated ProtocolLens data and artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.provenance import file_sha256  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv"
    retrieval_dir = PROJECT_ROOT / "artifacts" / "retrieval"
    model_dir = PROJECT_ROOT / "artifacts" / "models"
    frame = pd.read_csv(data_path)
    require(len(frame) >= 1_000, "Expected at least 1,000 development trials")
    require(frame["nct_id"].is_unique, "NCT identifiers must be unique")
    require(frame["query_conditions"].nunique() >= 3, "Expected at least three areas")
    counts = frame["overall_status"].value_counts()
    require(counts.get("COMPLETED", 0) >= 400, "Too few completed trials")
    require(counts.get("TERMINATED", 0) >= 400, "Too few terminated trials")
    require(
        frame["eligibility_criteria"].notna().mean() >= 0.95,
        "Eligibility-text completeness is below 95%",
    )
    require(frame["complexity_index"].between(0, 100).all(), "Invalid complexity index")

    run_manifest = json.loads(
        (PROJECT_ROOT / "artifacts" / "run_manifest.json").read_text(encoding="utf-8")
    )
    require(run_manifest["schema_version"] == 1, "Unexpected run-manifest schema")
    for relative_path, evidence in run_manifest["files"].items():
        artifact_path = PROJECT_ROOT / relative_path
        require(artifact_path.exists(), f"Manifest artifact missing: {relative_path}")
        require(
            file_sha256(artifact_path) == evidence["sha256"],
            f"Manifest hash mismatch: {relative_path}",
        )

    retrieval_metadata = json.loads((retrieval_dir / "metadata.json").read_text(encoding="utf-8"))
    require(retrieval_metadata["records"] == len(frame), "Retrieval/data row mismatch")
    require(
        retrieval_metadata["embedding_dimensions"] == 384,
        "Unexpected sentence embedding width",
    )
    proxy = json.loads((retrieval_dir / "proxy_evaluation.json").read_text(encoding="utf-8"))
    require(set(proxy["metrics"]) == {"tfidf", "semantic"}, "Proxy methods missing")
    text_leakage_audit = json.loads(
        (model_dir / "text_leakage_audit.json").read_text(encoding="utf-8")
    )
    require(
        text_leakage_audit["sanitized_model_text"]["records_with_any_term"] == 0,
        "Outcome-language sanitization failed",
    )
    review = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation.csv")
    require(len(review) == 60, "Retrieval evaluation sheet should contain 60 result rows")
    require(set(review["method"]) == {"tfidf", "semantic"}, "Review methods missing")
    unique_review_pairs = int(
        review[["query_nct_id", "candidate_nct_id"]].drop_duplicates().shape[0]
    )
    review_context_columns = {
        "query_phase",
        "query_intervention_types",
        "query_eligibility",
        "candidate_phase",
        "candidate_intervention_types",
        "candidate_eligibility",
    }
    require(
        review_context_columns.issubset(review.columns),
        "Retrieval review protocol context is incomplete",
    )
    require(
        review[["query_eligibility", "candidate_eligibility"]].notna().all().all(),
        "Retrieval review eligibility context contains blanks",
    )
    labelled_judgments = int(pd.to_numeric(review["relevance_0_2"], errors="coerce").notna().sum())
    evaluation_complete = labelled_judgments == len(review)
    evaluation_type = "pending"
    independent_human_review = False
    if evaluation_complete:
        require(
            (retrieval_dir / "manual_evaluation.json").exists(),
            "Run score_retrieval_evaluation.py after completing judgments",
        )
        relevance_evaluation = json.loads(
            (retrieval_dir / "manual_evaluation.json").read_text(encoding="utf-8")
        )
        evaluation_type = relevance_evaluation["review_type"]
        independent_human_review = bool(relevance_evaluation.get("independent_human_review", False))
        review_sources = (
            set(review["review_source"].dropna().astype(str))
            if "review_source" in review.columns
            else set()
        )
        if review_sources == {"ai_assisted_llm"}:
            require(
                relevance_evaluation.get("review_source") == "ai_assisted_llm"
                and not independent_human_review,
                "AI-assisted review provenance is missing or misleading",
            )

    comparison = pd.read_csv(model_dir / "model_comparison.csv")
    selection_comparison = pd.read_csv(model_dir / "selection_comparison.csv")
    # The interview edition intentionally compares one interpretable algorithm
    # across three feature representations.
    expected_models = {
        f"logistic_regression__{mode}" for mode in ("structured", "nlp", "combined")
    }
    require(expected_models.issubset(set(comparison["model"])), "Model variants incomplete")
    require("dummy_prior__baseline" in set(comparison["model"]), "Prior baseline missing")
    require(
        set(selection_comparison["model"]) == expected_models,
        "Validation model variants incomplete",
    )
    require(
        comparison[["roc_auc", "average_precision", "brier_score"]].notna().all().all(),
        "Model comparison contains missing primary metrics",
    )
    summary = json.loads((model_dir / "training_summary.json").read_text(encoding="utf-8"))
    require(summary["records_train"] > summary["records_test"], "Invalid split sizes")
    require(summary["records_validation"] == summary["records_test"], "Unexpected split sizes")
    require(summary["missing_date_records_excluded"] == 0, "Unexpected missing dates")
    require(
        (model_dir / "global_feature_importance.csv").exists(),
        "Explainability artifact missing",
    )
    require((model_dir / "subgroup_metrics.csv").exists(), "Subgroup metrics missing")
    require((model_dir / "error_analysis.csv").exists(), "Error analysis missing")
    validation_comparison = json.loads(
        (model_dir / "validation_comparison.json").read_text(encoding="utf-8")
    )
    require(
        set(validation_comparison) >= {"chronological", "random_stratified"},
        "Validation-strategy comparison missing",
    )
    result = {
        "status": "PASS" if evaluation_complete else "PASS_WITH_RETRIEVAL_REVIEW_PENDING",
        "trials": len(frame),
        "class_counts": counts.to_dict(),
        "therapeutic_areas": int(frame["query_conditions"].nunique()),
        "model_variants": len(expected_models),
        "additional_baselines": 1,
        "provenance_files_verified": len(run_manifest["files"]),
        "raw_outcome_language_matches": text_leakage_audit["raw"]["records_with_any_term"],
        "sanitized_model_text_matches": text_leakage_audit["sanitized_model_text"][
            "records_with_any_term"
        ],
        "best_model": summary["best_uncalibrated_model"],
        "retrieval_judgments_prepared": len(review),
        "retrieval_unique_comparisons": unique_review_pairs,
        "retrieval_judgments_completed": labelled_judgments,
        "retrieval_evaluation_type": evaluation_type,
        "independent_human_review": independent_human_review,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
