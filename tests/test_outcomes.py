import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocollens.outcomes import categorize_termination_reason


class OutcomeDescriptionTests(unittest.TestCase):
    def test_reason_categories(self):
        self.assertEqual(
            categorize_termination_reason("Slow participant recruitment"),
            "Recruitment or enrollment",
        )
        self.assertEqual(
            categorize_termination_reason("Terminated because of toxicity"),
            "Safety or tolerability",
        )
        self.assertEqual(categorize_termination_reason(None), "Not reported")


if __name__ == "__main__":
    unittest.main()
