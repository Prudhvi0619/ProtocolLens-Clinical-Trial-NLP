"""Compare a random split with the primary chronological holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.modelling import (  # noqa: E402
    TARGET_COLUMN,
    evaluate_probabilities,
    make_model_pipeline,
    prepare_model_frame,
    temporal_split,
)


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    model = make_model_pipeline("logistic_regression", "nlp")
    model.fit(train.drop(columns=[TARGET_COLUMN]), train[TARGET_COLUMN])
    probabilities = model.predict_proba(test.drop(columns=[TARGET_COLUMN]))[:, 1]
    return evaluate_probabilities(test[TARGET_COLUMN].to_numpy(), probabilities)


def main() -> None:
    frame = prepare_model_frame(
        pd.read_csv(PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv")
    )
    temporal = temporal_split(frame, 0.2)
    temporal_train = frame.loc[temporal.train_indices]
    temporal_test = frame.loc[temporal.test_indices]
    random_train, random_test = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame[TARGET_COLUMN],
    )
    result = {
        "model": "logistic_regression__nlp",
        "warning": (
            "Random splitting is a secondary sensitivity analysis. The chronological "
            "holdout remains the primary estimate for future-record generalization."
        ),
        "chronological": {
            "train_records": len(temporal_train),
            "test_records": len(temporal_test),
            "cutoff": temporal.cutoff_date,
            **fit_and_score(temporal_train, temporal_test),
        },
        "random_stratified": {
            "train_records": len(random_train),
            "test_records": len(random_test),
            **fit_and_score(random_train, random_test),
        },
    }
    output = PROJECT_ROOT / "artifacts" / "models" / "validation_comparison.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
