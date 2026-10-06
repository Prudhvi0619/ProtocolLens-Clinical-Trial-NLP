"""Run leakage-conscious structured, NLP, and combined model experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Keep matplotlib's cache inside the project rather than the user profile.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402

from protocollens.modelling import (  # noqa: E402
    TARGET_COLUMN,
    bootstrap_metric_interval,
    evaluate_probabilities,
    global_feature_importance,
    make_model_pipeline,
    paired_bootstrap_difference,
    prepare_model_frame,
    reliability_table,
    temporal_three_way_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all Logistic Regression feature variants")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "models")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Build a model-ready table, then split chronologically so future trials do
    # not influence preprocessing or model selection for earlier trials.
    frame = prepare_model_frame(pd.read_csv(args.input))
    split = temporal_three_way_split(
        frame,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    selection_train = frame.loc[split.train_indices]
    validation = frame.loc[split.validation_indices]
    development = frame.loc[list(split.train_indices) + list(split.validation_indices)].sort_values(
        ["validation_date", "nct_id"]
    )
    test = frame.loc[split.test_indices]
    x_selection_train = selection_train.drop(columns=[TARGET_COLUMN])
    y_selection_train = selection_train[TARGET_COLUMN].to_numpy()
    x_validation = validation.drop(columns=[TARGET_COLUMN])
    y_validation = validation[TARGET_COLUMN].to_numpy()
    x_development = development.drop(columns=[TARGET_COLUMN])
    y_development = development[TARGET_COLUMN].to_numpy()
    x_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN].to_numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Compare the same Logistic Regression algorithm across three feature sets.
    # This isolates the value added by eligibility text from algorithm choice.
    model_specs = [
        ("logistic_regression", mode) for mode in ("structured", "nlp", "combined")
    ]
    selection_rows: list[dict] = []
    print("Model selection on middle chronological partition", flush=True)
    for family, mode in model_specs:
        name = f"{family}__{mode}"
        model = make_model_pipeline(family, mode)
        model.fit(x_selection_train, y_selection_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        selection_rows.append(
            {
                "model": name,
                "family": family,
                "mode": mode,
                **evaluate_probabilities(y_validation, probabilities),
            }
        )
    selection_comparison = pd.DataFrame(selection_rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    selection_comparison.to_csv(args.output_dir / "selection_comparison.csv", index=False)
    # Select only on the middle chronological partition; the latest partition
    # remains untouched until the final evaluation below.
    selected_name = str(selection_comparison.iloc[0]["model"])
    selected_family = str(selection_comparison.iloc[0]["family"])
    selected_mode = str(selection_comparison.iloc[0]["mode"])
    print(f"Selected on validation: {selected_name}", flush=True)

    rows: list[dict] = []
    fitted: dict[str, object] = {}
    prediction_table = test[["nct_id", "overall_status", "validation_date"]].copy()
    model_probabilities: dict[str, object] = {}
    print("Final evaluation on latest chronological partition", flush=True)
    for family, mode in model_specs:
        name = f"{family}__{mode}"
        print(f"Training {name} ...", flush=True)
        model = make_model_pipeline(family, mode)
        model.fit(x_development, y_development)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = evaluate_probabilities(y_test, probabilities)
        ap_lower, ap_upper = bootstrap_metric_interval(y_test, probabilities, "average_precision")
        auc_lower, auc_upper = bootstrap_metric_interval(y_test, probabilities, "roc_auc")
        rows.append(
            {
                "model": name,
                "family": family,
                "mode": mode,
                **metrics,
                "average_precision_ci_lower": ap_lower,
                "average_precision_ci_upper": ap_upper,
                "roc_auc_ci_lower": auc_lower,
                "roc_auc_ci_upper": auc_upper,
            }
        )
        fitted[name] = model
        model_probabilities[name] = probabilities
        prediction_table[name] = probabilities
        print(
            f"  ROC-AUC={metrics['roc_auc']:.3f} "
            f"AP={metrics['average_precision']:.3f} "
            f"Brier={metrics['brier_score']:.3f}",
            flush=True,
        )

    baseline_name = "dummy_prior__baseline"
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(x_development, y_development)
    baseline_probabilities = baseline.predict_proba(x_test)[:, 1]
    baseline_metrics = evaluate_probabilities(y_test, baseline_probabilities)
    baseline_ap_lower, baseline_ap_upper = bootstrap_metric_interval(
        y_test, baseline_probabilities, "average_precision"
    )
    baseline_auc_lower, baseline_auc_upper = bootstrap_metric_interval(
        y_test, baseline_probabilities, "roc_auc"
    )
    rows.append(
        {
            "model": baseline_name,
            "family": "baseline",
            "mode": "prior",
            **baseline_metrics,
            "average_precision_ci_lower": baseline_ap_lower,
            "average_precision_ci_upper": baseline_ap_upper,
            "roc_auc_ci_lower": baseline_auc_lower,
            "roc_auc_ci_upper": baseline_auc_upper,
        }
    )
    prediction_table[baseline_name] = baseline_probabilities

    comparison = pd.DataFrame(rows).sort_values(["average_precision", "roc_auc"], ascending=False)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    prediction_table.to_csv(args.output_dir / "all_test_predictions.csv", index=False)
    best_name = selected_name
    best_uncalibrated = fitted[selected_name]
    joblib.dump(best_uncalibrated, args.output_dir / "best_uncalibrated.joblib")

    print(f"Calibrating validation-selected model: {best_name} ...", flush=True)
    calibrated = CalibratedClassifierCV(
        estimator=make_model_pipeline(selected_family, selected_mode),
        method="sigmoid",
        cv=3,
    )
    calibrated.fit(x_development, y_development)
    calibrated_probabilities = calibrated.predict_proba(x_test)[:, 1]
    calibrated_metrics = evaluate_probabilities(y_test, calibrated_probabilities)
    joblib.dump(calibrated, args.output_dir / "best_calibrated.joblib")
    reliability_table(y_test, calibrated_probabilities).to_csv(
        args.output_dir / "reliability.csv", index=False
    )
    prediction_rows = test[["nct_id", "overall_status", "validation_date"]].copy()
    prediction_rows["probability_terminated"] = calibrated_probabilities
    prediction_rows.to_csv(args.output_dir / "test_predictions.csv", index=False)

    # Quantify whether adding eligibility NLP improves over registry fields.
    nlp_value_add = {
        "logistic_regression": paired_bootstrap_difference(
            y_test,
            model_probabilities["logistic_regression__nlp"],
            model_probabilities["logistic_regression__structured"],
        )
    }

    importance = global_feature_importance(best_uncalibrated)
    importance.to_csv(args.output_dir / "global_feature_importance.csv", index=False)
    summary = {
        "records_total": len(frame),
        "records_selection_train": len(selection_train),
        "records_validation": len(validation),
        "records_train": len(development),
        "records_test": len(test),
        "selection_train_class_counts": selection_train["overall_status"].value_counts().to_dict(),
        "validation_class_counts": validation["overall_status"].value_counts().to_dict(),
        "train_class_counts": development["overall_status"].value_counts().to_dict(),
        "test_class_counts": test["overall_status"].value_counts().to_dict(),
        "validation_cutoff": split.validation_cutoff_date,
        "temporal_cutoff": split.test_cutoff_date,
        "missing_date_records_excluded": split.excluded_missing_dates,
        "best_uncalibrated_model": best_name,
        "selection_basis": "highest validation average precision, then ROC-AUC",
        "selected_model_validation_metrics": selection_comparison.iloc[0]
        .drop(labels=["model", "family", "mode"])
        .to_dict(),
        "best_uncalibrated_metrics": comparison.loc[comparison["model"] == selected_name]
        .iloc[0]
        .drop(labels=["model", "family", "mode"])
        .to_dict(),
        "highest_test_metric_model_descriptive_only": str(comparison.iloc[0]["model"]),
        "prior_baseline_test_metrics": baseline_metrics,
        "calibration_method": "sigmoid, 3-fold training-set cross-validation",
        "calibrated_test_metrics": calibrated_metrics,
        "nlp_vs_structured_average_precision": nlp_value_add,
        "probability_scope": (
            "Scores are estimated from this deliberately balanced retrospective sample "
            "and do not represent real-world trial termination probabilities."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
