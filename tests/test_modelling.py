import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocollens.modelling import (
    LEAKAGE_BLOCKLIST,
    NLP_NUMERIC,
    STRUCTURED_CATEGORICAL,
    STRUCTURED_NUMERIC,
    TARGET_COLUMN,
    age_to_years,
    audit_outcome_language,
    bootstrap_metric_interval,
    evaluate_probabilities,
    explain_linear_prediction,
    make_model_pipeline,
    prepare_model_frame,
    sanitize_outcome_language,
    temporal_split,
    temporal_three_way_split,
)


class ModelPreparationTests(unittest.TestCase):
    def test_leakage_fields_are_not_selected_features(self):
        selected = set(STRUCTURED_CATEGORICAL + STRUCTURED_NUMERIC + NLP_NUMERIC)
        self.assertTrue(selected.isdisjoint(LEAKAGE_BLOCKLIST))

    def test_outcome_language_audit_detects_exact_terms(self):
        audit = audit_outcome_language(
            pd.Series(["Trial was terminated", "Patients completed therapy", "No signal"])
        )
        self.assertEqual(audit["records_with_any_term"], 2)
        self.assertEqual(audit["terms"]["terminated"], 1)
        self.assertEqual(audit["terms"]["completed"], 1)

    def test_outcome_language_is_removed_before_nlp(self):
        sanitized = sanitize_outcome_language(
            pd.Series(["Trial terminated after patients completed therapy"])
        )
        self.assertEqual(audit_outcome_language(sanitized)["records_with_any_term"], 0)

    def test_age_conversion(self):
        self.assertEqual(age_to_years("18 Years"), 18)
        self.assertAlmostEqual(age_to_years("6 Months"), 0.5)

    def test_temporal_split_is_chronological(self):
        rows = []
        for index in range(30):
            rows.append(
                {
                    "nct_id": f"NCT{index:03d}",
                    "overall_status": "TERMINATED" if index % 2 else "COMPLETED",
                    "study_first_submit_date": f"{2000 + index}-01-01",
                    "start_date": "",
                    "minimum_age": "18 Years",
                    "maximum_age": "65 Years",
                    "eligibility_criteria": "Adults",
                    "query_conditions": "Example",
                    "study_type": "INTERVENTIONAL",
                    "phases": "PHASE2",
                    "intervention_types": "DRUG",
                    "lead_sponsor_class": "OTHER",
                    "allocation": "RANDOMIZED",
                    "intervention_model": "PARALLEL",
                    "primary_purpose": "TREATMENT",
                    "masking": "NONE",
                    "sex": "ALL",
                    "healthy_volunteers": "No",
                    **{
                        column: 1
                        for column in (
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
                        )
                    },
                }
            )
        prepared = prepare_model_frame(pd.DataFrame(rows))
        split = temporal_split(prepared, 0.2)
        train = prepared.loc[split.train_indices]
        test = prepared.loc[split.test_indices]
        self.assertLess(train.validation_date.max(), test.validation_date.min())
        self.assertIn(TARGET_COLUMN, prepared)

        three_way = temporal_three_way_split(prepared)
        three_train = prepared.loc[three_way.train_indices]
        three_validation = prepared.loc[three_way.validation_indices]
        three_test = prepared.loc[three_way.test_indices]
        self.assertLess(three_train.validation_date.max(), three_validation.validation_date.min())
        self.assertLess(three_validation.validation_date.max(), three_test.validation_date.min())

    def test_probability_metrics_are_bounded(self):
        metrics = evaluate_probabilities(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.7, 0.9]))
        for value in metrics.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

    def test_bootstrap_interval_contains_good_observed_score(self):
        y_true = np.array([0, 0, 0, 1, 1, 1] * 10)
        probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10)
        lower, upper = bootstrap_metric_interval(y_true, probabilities, "roc_auc", iterations=100)
        self.assertLessEqual(lower, upper)
        self.assertGreater(lower, 0.9)

    def test_local_linear_explanation_has_signed_contributions(self):
        rows = []
        labels = []
        for index in range(8):
            labels.append(index % 2)
            rows.append(
                {
                    "eligibility_criteria": (
                        "strict oncology laboratory criteria"
                        if index % 2
                        else "simple adult diabetes criteria"
                    ),
                    **{column: index + 1 for column in NLP_NUMERIC},
                }
            )
        frame = pd.DataFrame(rows)
        model = make_model_pipeline("logistic_regression", "nlp")
        model.fit(frame, np.asarray(labels))
        explanation = explain_linear_prediction(model, frame.iloc[[0]], top_n=5)
        self.assertFalse(explanation.empty)
        self.assertIn("contribution", explanation)

    def test_nlp_pipeline_sanitizes_training_and_inference_text(self):
        rows = []
        labels = []
        for index in range(8):
            labels.append(index % 2)
            rows.append(
                {
                    "eligibility_criteria": (
                        "terminated marker oncology criteria"
                        if index % 2
                        else "completed marker diabetes criteria"
                    ),
                    **{column: index + 1 for column in NLP_NUMERIC},
                }
            )
        frame = pd.DataFrame(rows)
        model = make_model_pipeline("logistic_regression", "nlp")
        model.fit(frame, np.asarray(labels))
        text_pipeline = model.named_steps["preprocessor"].named_transformers_["eligibility_tfidf"]
        vocabulary = text_pipeline.named_steps["tfidf"].vocabulary_
        self.assertNotIn("terminated", vocabulary)
        self.assertNotIn("completed", vocabulary)

        raw = frame.iloc[[0]].copy()
        cleaned = raw.copy()
        cleaned["eligibility_criteria"] = "marker diabetes criteria"
        self.assertAlmostEqual(
            model.predict_proba(raw)[0, 1],
            model.predict_proba(cleaned)[0, 1],
        )


if __name__ == "__main__":
    unittest.main()
