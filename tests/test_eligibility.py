import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocollens.eligibility import eligibility_features, parse_eligibility

TEXT = """Inclusion Criteria:
* Adults aged 18 years or older
* HbA1c greater than 7.0% within 3 months

Exclusion Criteria:
1. No prior insulin treatment
2. Creatinine >= 2.0 mg/dL
"""


class EligibilityParserTests(unittest.TestCase):
    def test_assigns_sections(self):
        criteria = parse_eligibility(TEXT)
        self.assertEqual(len(criteria), 4)
        self.assertEqual(
            [item.section for item in criteria],
            ["inclusion", "inclusion", "exclusion", "exclusion"],
        )

    def test_extracts_transparent_features(self):
        features = eligibility_features(TEXT)
        self.assertEqual(features["inclusion_count"], 2)
        self.assertEqual(features["exclusion_count"], 2)
        self.assertGreaterEqual(features["numeric_constraint_count"], 4)
        self.assertGreaterEqual(features["temporal_constraint_count"], 2)
        self.assertGreater(features["complexity_index"], 0)

    def test_handles_missing_text(self):
        self.assertEqual(parse_eligibility(None), [])
        self.assertEqual(eligibility_features(None)["criterion_count"], 0)

    def test_complexity_index_is_capped(self):
        very_long = "Inclusion Criteria:\n" + "\n".join(
            f"* HbA1c >= {index} within {index} months" for index in range(100)
        )
        score = eligibility_features(very_long)["complexity_index"]
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)


if __name__ == "__main__":
    unittest.main()
