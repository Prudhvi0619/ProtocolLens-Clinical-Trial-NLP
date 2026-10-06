import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocollens.review import (
    apply_pair_judgment,
    review_progress,
    save_review_progress,
)


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            [
                {
                    "query_nct_id": "Q1",
                    "candidate_nct_id": "C1",
                    "method": "tfidf",
                    "relevance_0_2": float("nan"),
                    "reviewer_note": "",
                },
                {
                    "query_nct_id": "Q1",
                    "candidate_nct_id": "C1",
                    "method": "semantic",
                    "relevance_0_2": float("nan"),
                    "reviewer_note": "",
                },
                {
                    "query_nct_id": "Q1",
                    "candidate_nct_id": "C2",
                    "method": "semantic",
                    "relevance_0_2": float("nan"),
                    "reviewer_note": "",
                },
            ]
        )

    def test_shared_pair_judgment_propagates_to_both_methods(self):
        updated = apply_pair_judgment(self.frame, "Q1", "C1", 2, "close match")
        pair = updated[updated["candidate_nct_id"] == "C1"]
        self.assertEqual(pair["relevance_0_2"].tolist(), [2.0, 2.0])
        self.assertEqual(pair["reviewer_note"].tolist(), ["close match", "close match"])
        self.assertEqual(review_progress(updated), (2, 1, 2))

    def test_rejects_invalid_rating(self):
        with self.assertRaises(ValueError):
            apply_pair_judgment(self.frame, "Q1", "C1", 3)

    def test_atomic_save_round_trip(self):
        updated = apply_pair_judgment(self.frame, "Q1", "C2", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            save_review_progress(updated, path)
            loaded = pd.read_csv(path)
        self.assertEqual(int(loaded["relevance_0_2"].notna().sum()), 1)
        self.assertFalse(path.with_suffix(".csv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
