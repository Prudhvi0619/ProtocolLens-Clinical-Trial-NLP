import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ARTIFACTS_EXIST = all(
    [
        (ROOT / "data" / "processed" / "trials_with_features.csv").exists(),
        (ROOT / "artifacts" / "retrieval" / "tfidf.joblib").exists(),
        (ROOT / "artifacts" / "models" / "training_summary.json").exists(),
    ]
)


@unittest.skipUnless(
    INTEGRATION_ARTIFACTS_EXIST,
    "Generated data/model artifacts are not present in a clean clone",
)
class DashboardSmokeTests(unittest.TestCase):
    def test_all_pages_render_without_exception(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(app.exception, [])
        self.assertIn("ProtocolLens", [title.value for title in app.title])
        for page in [
            "Eligibility analyzer",
            "Comparable trials",
            "Retrieval review",
            "Model evidence",
            "Method & limitations",
        ]:
            with self.subTest(page=page):
                app.sidebar.radio[0].set_value(page).run(timeout=30)
                self.assertEqual(app.exception, [])
                if page == "Eligibility analyzer":
                    app.button[0].click().run(timeout=60)
                    self.assertEqual(app.exception, [])
                    self.assertTrue(
                        any("Termination-associated" in metric.label for metric in app.metric)
                    )


if __name__ == "__main__":
    unittest.main()
