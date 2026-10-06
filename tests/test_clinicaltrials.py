import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocollens.clinicaltrials import ClinicalTrialsClient, FetchConfig, normalize_study

SAMPLE_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000001",
            "briefTitle": "Example trial",
        },
        "statusModule": {
            "overallStatus": "TERMINATED",
            "whyStopped": "Slow enrollment",
            "startDateStruct": {"date": "2020-01"},
        },
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE2"],
            "enrollmentInfo": {"count": 75, "type": "ACTUAL"},
            "designInfo": {"allocation": "RANDOMIZED"},
        },
        "eligibilityModule": {
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "70 Years",
            "eligibilityCriteria": "Inclusion Criteria:\n* Age 18 or older",
        },
        "conditionsModule": {"conditions": ["Breast Cancer"]},
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "Example University", "class": "OTHER"}
        },
        "armsInterventionsModule": {
            "interventions": [
                {"type": "DRUG"},
                {"type": "DRUG"},
                {"type": "PROCEDURE"},
            ]
        },
        "contactsLocationsModule": {
            "locations": [
                {"country": "United States"},
                {"country": "United States"},
                {"country": "Canada"},
            ]
        },
    }
}


class FetchConfigTests(unittest.TestCase):
    def test_rejects_blank_condition(self):
        with self.assertRaises(ValueError):
            FetchConfig(condition=" ")

    def test_rejects_invalid_page_size(self):
        with self.assertRaises(ValueError):
            FetchConfig(condition="Cancer", page_size=1001)


class PaginationTests(unittest.TestCase):
    def test_iterates_pages_and_respects_maximum(self):
        class FakeClient(ClinicalTrialsClient):
            def __init__(self):
                super().__init__("https://example.invalid")
                self.calls = []

            def _get_json(self, path, params, config):
                self.calls.append(dict(params))
                if "pageToken" not in params:
                    return {"studies": [{"id": 1}, {"id": 2}], "nextPageToken": "next"}
                return {"studies": [{"id": 3}, {"id": 4}]}

        client = FakeClient()
        studies = list(
            client.iter_studies(FetchConfig(condition="Cancer", max_studies=3, page_size=2))
        )
        self.assertEqual([study["id"] for study in studies], [1, 2, 3])
        self.assertEqual(client.calls[1]["pageToken"], "next")
        self.assertEqual(client.calls[0]["filter.overallStatus"], "COMPLETED|TERMINATED")


class NormalizationTests(unittest.TestCase):
    def test_normalizes_nested_study(self):
        row = normalize_study(SAMPLE_STUDY)
        self.assertEqual(row["nct_id"], "NCT00000001")
        self.assertEqual(row["overall_status"], "TERMINATED")
        self.assertEqual(row["intervention_types"], "DRUG|PROCEDURE")
        self.assertEqual(row["number_of_locations"], 3)
        self.assertEqual(row["countries"], "United States|Canada")
        self.assertIn("Age 18", row["eligibility_criteria"])


if __name__ == "__main__":
    unittest.main()
