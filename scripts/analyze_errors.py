"""Generate holdout subgroup metrics and inspect high-confidence errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"


def main() -> None:
    summary = json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8"))
    best_model = summary["best_uncalibrated_model"]
    predictions = pd.read_csv(MODEL_DIR / "all_test_predictions.csv")
    trials = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv")
    columns = [
        "nct_id",
        "brief_title",
        "query_conditions",
        "phases",
        "complexity_index",
        "criterion_count",
    ]
    analysis = predictions.merge(trials[columns], on="nct_id", how="left")
    analysis["y_true"] = (analysis["overall_status"] == "TERMINATED").astype(int)
    analysis["predicted_terminated"] = (analysis[best_model] >= 0.5).astype(int)
    analysis["error_type"] = np.select(
        [
            (analysis["y_true"] == 0) & (analysis["predicted_terminated"] == 1),
            (analysis["y_true"] == 1) & (analysis["predicted_terminated"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    analysis["confidence"] = np.where(
        analysis["predicted_terminated"] == 1,
        analysis[best_model],
        1 - analysis[best_model],
    )
    errors = analysis[analysis["error_type"] != "correct"].sort_values(
        "confidence", ascending=False
    )
    errors.to_csv(MODEL_DIR / "error_analysis.csv", index=False)

    subgroup_rows = []
    for area, group in analysis.groupby("query_conditions"):
        y_true = group["y_true"].to_numpy()
        probability = group[best_model].to_numpy()
        predicted = group["predicted_terminated"].to_numpy()
        false_positive = int(np.sum((y_true == 0) & (predicted == 1)))
        false_negative = int(np.sum((y_true == 1) & (predicted == 0)))
        negatives = int(np.sum(y_true == 0))
        positives = int(np.sum(y_true == 1))
        subgroup_rows.append(
            {
                "query_conditions": area,
                "records": len(group),
                "terminated_fraction": float(np.mean(y_true)),
                "average_precision": average_precision_score(y_true, probability),
                "roc_auc": roc_auc_score(y_true, probability),
                "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
                "false_positive_rate": false_positive / negatives if negatives else np.nan,
                "false_negative_rate": false_negative / positives if positives else np.nan,
            }
        )
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(MODEL_DIR / "subgroup_metrics.csv", index=False)
    print(f"Best model: {best_model}")
    print(analysis["error_type"].value_counts().to_string())
    print("\nSubgroup metrics:")
    print(subgroup.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
