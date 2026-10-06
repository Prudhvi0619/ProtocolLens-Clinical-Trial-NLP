"""Leakage-conscious model preparation, evaluation, and explainability."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ModelMode = Literal["structured", "nlp", "combined"]
TARGET_COLUMN = "target_terminated"
TEXT_COLUMN = "eligibility_criteria"

STRUCTURED_CATEGORICAL = [
    "query_conditions",
    "study_type",
    "phases",
    "intervention_types",
    "lead_sponsor_class",
    "allocation",
    "intervention_model",
    "primary_purpose",
    "masking",
    "sex",
    "healthy_volunteers",
]
STRUCTURED_NUMERIC = [
    "minimum_age_years",
    "maximum_age_years",
    "submission_year",
]
NLP_NUMERIC = [
    "eligibility_word_count",
    "criterion_count",
    "inclusion_count",
    "exclusion_count",
    "average_criterion_words",
    "exclusion_ratio",
    "numeric_constraint_count",
    "comparator_count",
    "temporal_constraint_count",
    "negation_count",
    "clinical_measure_count",
    "complexity_index",
]

# These fields are intentionally excluded from every feature pipeline. Several
# are known only after or during trial conduct and could make results look much
# better without being prospectively useful.
LEAKAGE_BLOCKLIST = [
    "overall_status",
    "why_stopped",
    "completion_date",
    "last_update_post_date",
    "enrollment_count",
    "enrollment_type",
    "number_of_locations",
]

OUTCOME_LANGUAGE_TERMS = (
    "terminated",
    "termination",
    "completed",
    "completion",
    "stopped",
    "withdrawn",
    "suspended",
)
OUTCOME_LANGUAGE_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, OUTCOME_LANGUAGE_TERMS)) + r")\b",
    flags=re.IGNORECASE,
)


def audit_outcome_language(texts: pd.Series) -> dict[str, Any]:
    """Count explicit registry-outcome words in candidate model text."""
    normalized = texts.fillna("").astype(str).str.lower()
    term_counts = {
        term: int(normalized.str.contains(rf"\b{re.escape(term)}\b", regex=True).sum())
        for term in OUTCOME_LANGUAGE_TERMS
    }
    return {
        "terms": term_counts,
        "records_with_any_term": int(
            normalized.str.contains(OUTCOME_LANGUAGE_PATTERN, regex=True).sum()
        ),
    }


def sanitize_outcome_language(texts: Any) -> pd.Series:
    """Remove exact outcome-language tokens before TF-IDF fitting or inference."""
    normalized = pd.Series(texts).fillna("").astype(str)
    return normalized.str.replace(OUTCOME_LANGUAGE_PATTERN, " ", regex=True)


def age_to_years(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    match = re.search(
        r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>years?|months?|weeks?|days?)",
        str(value),
        re.IGNORECASE,
    )
    if not match:
        return np.nan
    number = float(match.group("number"))
    unit = match.group("unit").lower()
    if unit.startswith("month"):
        return number / 12
    if unit.startswith("week"):
        return number / 52.1429
    if unit.startswith("day"):
        return number / 365.25
    return number


def prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only predeclared model fields and a chronological validation date."""
    prepared = frame.copy()
    prepared = prepared[prepared["overall_status"].isin(["COMPLETED", "TERMINATED"])]
    prepared[TARGET_COLUMN] = (prepared["overall_status"] == "TERMINATED").astype(int)
    prepared["minimum_age_years"] = prepared["minimum_age"].map(age_to_years)
    prepared["maximum_age_years"] = prepared["maximum_age"].map(age_to_years)
    submit_date = pd.to_datetime(
        prepared["study_first_submit_date"], errors="coerce", format="mixed"
    )
    if "study_first_submit_qc_date" in prepared:
        submit_qc = pd.to_datetime(
            prepared["study_first_submit_qc_date"], errors="coerce", format="mixed"
        )
        submit_date = submit_date.fillna(submit_qc)
    start_date = pd.to_datetime(prepared["start_date"], errors="coerce", format="mixed")
    prepared["validation_date"] = submit_date.fillna(start_date)
    prepared["submission_year"] = prepared["validation_date"].dt.year
    prepared[TEXT_COLUMN] = prepared[TEXT_COLUMN].fillna("").astype(str)
    for column in STRUCTURED_CATEGORICAL:
        prepared[column] = prepared[column].fillna("MISSING").astype(str)
    for column in NLP_NUMERIC:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


@dataclass(frozen=True)
class TemporalSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    cutoff_date: str
    excluded_missing_dates: int


@dataclass(frozen=True)
class TemporalThreeWaySplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    validation_cutoff_date: str
    test_cutoff_date: str
    excluded_missing_dates: int


def temporal_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> TemporalSplit:
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")
    dated = frame[frame["validation_date"].notna()].sort_values(["validation_date", "nct_id"])
    if len(dated) < 20:
        raise ValueError("At least 20 dated trials are required")
    split_position = int(len(dated) * (1 - test_fraction))
    split_position = min(max(split_position, 1), len(dated) - 1)
    train = dated.iloc[:split_position]
    test = dated.iloc[split_position:]
    if train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Both temporal partitions must contain both classes")
    return TemporalSplit(
        train_indices=train.index.to_numpy(),
        test_indices=test.index.to_numpy(),
        cutoff_date=test["validation_date"].min().date().isoformat(),
        excluded_missing_dates=int(frame["validation_date"].isna().sum()),
    )


def temporal_three_way_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> TemporalThreeWaySplit:
    """Oldest train, middle validation, newest test partitions."""
    if validation_fraction < 0.1 or test_fraction < 0.1:
        raise ValueError("validation and test fractions must each be at least 0.1")
    if validation_fraction + test_fraction > 0.5:
        raise ValueError("validation and test fractions together must not exceed 0.5")
    dated = frame[frame["validation_date"].notna()].sort_values(["validation_date", "nct_id"])
    if len(dated) < 30:
        raise ValueError("At least 30 dated trials are required")
    train_end = int(len(dated) * (1 - validation_fraction - test_fraction))
    validation_end = int(len(dated) * (1 - test_fraction))
    train = dated.iloc[:train_end]
    validation = dated.iloc[train_end:validation_end]
    test = dated.iloc[validation_end:]
    for name, partition in (
        ("training", train),
        ("validation", validation),
        ("test", test),
    ):
        if partition[TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"{name} partition must contain both classes")
    return TemporalThreeWaySplit(
        train_indices=train.index.to_numpy(),
        validation_indices=validation.index.to_numpy(),
        test_indices=test.index.to_numpy(),
        validation_cutoff_date=validation["validation_date"].min().date().isoformat(),
        test_cutoff_date=test["validation_date"].min().date().isoformat(),
        excluded_missing_dates=int(frame["validation_date"].isna().sum()),
    )


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )


def _categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
            ),
        ]
    )


def make_preprocessor(mode: ModelMode, max_text_features: int = 5_000) -> ColumnTransformer:
    """Build the feature pipeline for the selected experiment.

    ``ColumnTransformer`` processes each feature family independently and then
    concatenates the resulting columns before Logistic Regression sees them.
    """
    transformers: list[tuple] = []
    if mode in ("structured", "combined"):
        # Registry numbers are imputed/scaled; registry categories are one-hot encoded.
        transformers.extend(
            [
                ("structured_numeric", _numeric_pipeline(), STRUCTURED_NUMERIC),
                (
                    "structured_categorical",
                    _categorical_pipeline(),
                    STRUCTURED_CATEGORICAL,
                ),
            ]
        )
    if mode in ("nlp", "combined"):
        # Join engineered eligibility counts with a sparse text representation.
        transformers.extend(
            [
                ("nlp_numeric", _numeric_pipeline(), NLP_NUMERIC),
                (
                    "eligibility_tfidf",
                    Pipeline(
                        [
                            (
                                "remove_outcome_language",
                                FunctionTransformer(
                                    sanitize_outcome_language,
                                    validate=False,
                                    feature_names_out="one-to-one",
                                ),
                            ),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    stop_words="english",
                                    ngram_range=(1, 2),
                                    min_df=3,
                                    max_features=max_text_features,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                    TEXT_COLUMN,
                ),
            ]
        )
    return ColumnTransformer(transformers, remainder="drop")


def make_model_pipeline(
    family: Literal["logistic_regression"],
    mode: ModelMode,
    random_state: int = 42,
) -> Pipeline:
    """Create the leakage-safe Logistic Regression pipeline used in experiments."""
    if family != "logistic_regression":
        raise ValueError(f"Unknown model family: {family}")
    preprocessor = make_preprocessor(mode)
    estimator = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )
    # Keeping preprocessing and estimation in one Pipeline ensures validation
    # and test rows are transformed only with statistics learned from training.
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1 else probabilities <= upper
        )
        if not np.any(mask):
            continue
        error += np.sum(mask) / total * abs(np.mean(y_true[mask]) - np.mean(probabilities[mask]))
    return float(error)


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "ece_10_bin": expected_calibration_error(y_true, probabilities),
    }


def reliability_table(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for number, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True), start=1):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1 else probabilities <= upper
        )
        rows.append(
            {
                "bin": number,
                "lower": lower,
                "upper": upper,
                "count": int(np.sum(mask)),
                "mean_predicted_probability": float(np.mean(probabilities[mask]))
                if np.any(mask)
                else np.nan,
                "observed_terminated_fraction": float(np.mean(y_true[mask]))
                if np.any(mask)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: Literal["average_precision", "roc_auc"],
    iterations: int = 500,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return a non-parametric 95% confidence interval on the holdout."""
    if iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        sampled_p = probabilities[indices]
        score = (
            average_precision_score(sampled_y, sampled_p)
            if metric == "average_precision"
            else roc_auc_score(sampled_y, sampled_p)
        )
        values.append(float(score))
    if not values:
        raise ValueError("Bootstrap samples did not contain both classes")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def paired_bootstrap_difference(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    iterations: int = 500,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate AP(A)-AP(B) on identical resampled holdout records."""
    rng = np.random.default_rng(random_state)
    differences: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        differences.append(
            float(
                average_precision_score(sampled_y, probabilities_a[indices])
                - average_precision_score(sampled_y, probabilities_b[indices])
            )
        )
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "mean_ap_difference": float(np.mean(differences)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "probability_difference_positive": float(np.mean(np.asarray(differences) > 0)),
    }


def global_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Rank Logistic Regression features by absolute coefficient magnitude."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]
    names = preprocessor.get_feature_names_out()
    if not hasattr(estimator, "coef_"):
        raise TypeError("Logistic Regression estimator does not expose coefficients")
    values = estimator.coef_[0]
    importance_type = "coefficient"
    result = pd.DataFrame(
        {
            "feature": names,
            "importance": values,
            "absolute_importance": np.abs(values),
            "importance_type": importance_type,
        }
    )
    return result.sort_values("absolute_importance", ascending=False).reset_index(drop=True)


def explain_linear_prediction(
    pipeline: Pipeline, frame: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Return per-row linear contributions for an fitted Logistic Regression pipeline."""
    if len(frame) != 1:
        raise ValueError("Exactly one row is required for a local explanation")
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]
    if not hasattr(estimator, "coef_"):
        raise TypeError("Local linear explanation requires an estimator with coefficients")
    transformed = preprocessor.transform(frame)
    coefficients = estimator.coef_[0]
    if hasattr(transformed, "multiply"):
        contributions = transformed.multiply(coefficients).toarray()[0]
    else:
        contributions = np.asarray(transformed)[0] * coefficients
    names = preprocessor.get_feature_names_out()
    explanation = pd.DataFrame(
        {
            "feature": names,
            "contribution": contributions,
            "absolute_contribution": np.abs(contributions),
        }
    )
    explanation = explanation[explanation["absolute_contribution"] > 0]
    return explanation.nlargest(top_n, "absolute_contribution").reset_index(drop=True)
