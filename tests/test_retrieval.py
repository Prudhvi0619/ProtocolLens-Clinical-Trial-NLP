import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocollens.retrieval import (
    SemanticTrialRetriever,
    TfidfTrialRetriever,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    select_review_query_ids,
    summarize_judgments,
)

RECORDS = [
    {
        "nct_id": "NCT1",
        "brief_title": "Diabetes metformin study",
        "conditions": "Type 2 Diabetes",
        "eligibility_criteria": "Adults with HbA1c above 7 percent",
    },
    {
        "nct_id": "NCT2",
        "brief_title": "Glucose control trial",
        "conditions": "Type 2 Diabetes",
        "eligibility_criteria": "Adults taking metformin with high HbA1c",
    },
    {
        "nct_id": "NCT3",
        "brief_title": "Breast cancer radiation study",
        "conditions": "Breast Cancer",
        "eligibility_criteria": "Women with localized breast carcinoma",
    },
]


class TfidfRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.retriever = TfidfTrialRetriever(RECORDS)

    def test_search_finds_related_trial(self):
        result = self.retriever.search("metformin diabetes HbA1c", k=1)[0]
        self.assertIn(result["nct_id"], {"NCT1", "NCT2"})

    def test_search_by_id_excludes_query(self):
        results = self.retriever.search_by_id("NCT1", k=2)
        self.assertNotIn("NCT1", [result["nct_id"] for result in results])
        self.assertEqual(results[0]["nct_id"], "NCT2")

    def test_explanation_returns_shared_weighted_terms(self):
        terms = self.retriever.explain_pair("NCT1", "NCT2", top_n=10)
        names = {item["term"] for item in terms}
        self.assertTrue({"metformin", "hba1c"} & names)


class SemanticRetrieverTests(unittest.TestCase):
    def test_precomputed_embeddings_rank_without_loading_model(self):
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        retriever = SemanticTrialRetriever(RECORDS, embeddings=embeddings)
        result = retriever.search_by_id("NCT1", k=1)[0]
        self.assertEqual(result["nct_id"], "NCT2")


class RetrievalMetricTests(unittest.TestCase):
    def test_metrics(self):
        relevances = [0, 2, 1]
        self.assertAlmostEqual(precision_at_k(relevances, 3), 2 / 3)
        self.assertEqual(reciprocal_rank(relevances), 0.5)
        self.assertGreater(ndcg_at_k(relevances, 3), 0)
        self.assertLessEqual(ndcg_at_k(relevances, 3), 1)

    def test_manual_summary_accepts_dashboard_float_strings(self):
        rows = [
            {
                "query_nct_id": "Q1",
                "method": "semantic",
                "rank": rank,
                "relevance_0_2": value,
            }
            for rank, value in enumerate(["2.0", "1.0", "0.0"], start=1)
        ]
        summary = summarize_judgments(rows, k=3)
        self.assertEqual(summary["semantic"]["precision_at_k"], 0.6667)

    def test_manual_summary_rejects_blank_judgment(self):
        with self.assertRaises(ValueError):
            summarize_judgments(
                [
                    {
                        "query_nct_id": "Q1",
                        "method": "tfidf",
                        "rank": 1,
                        "relevance_0_2": "",
                    }
                ]
            )

    def test_review_queries_balance_status_within_area(self):
        records = [
            {"nct_id": "A1", "query_conditions": "Area A", "overall_status": "COMPLETED"},
            {"nct_id": "A2", "query_conditions": "Area A", "overall_status": "TERMINATED"},
            {"nct_id": "B1", "query_conditions": "Area B", "overall_status": "COMPLETED"},
            {"nct_id": "B2", "query_conditions": "Area B", "overall_status": "TERMINATED"},
        ]
        self.assertEqual(select_review_query_ids(records), ["A1", "A2", "B1", "B2"])


if __name__ == "__main__":
    unittest.main()
