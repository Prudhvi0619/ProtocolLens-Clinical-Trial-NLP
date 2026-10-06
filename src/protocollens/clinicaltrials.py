"""ClinicalTrials.gov API ingestion and normalization.

This module intentionally uses only Python's standard library so that the data
pipeline can run before the modelling environment is installed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

API_BASE_URL = "https://clinicaltrials.gov/api/v2"
FINAL_STATUSES = ("COMPLETED", "TERMINATED")


def _dig(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely read a nested dictionary value."""
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _unique(values: Sequence[Any]) -> list[Any]:
    """Return non-empty values in their original order without duplicates."""
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value in (None, "") or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass(frozen=True)
class FetchConfig:
    condition: str
    max_studies: int = 500
    page_size: int = 100
    statuses: tuple[str, ...] = FINAL_STATUSES
    timeout_seconds: int = 30
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.condition.strip():
            raise ValueError("condition must not be empty")
        if self.max_studies < 1:
            raise ValueError("max_studies must be at least 1")
        if not 1 <= self.page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        if not self.statuses:
            raise ValueError("at least one status is required")


class ClinicalTrialsClient:
    """Small, retrying client for the public ClinicalTrials.gov v2 API."""

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _get_json(
        self, path: str, params: dict[str, str | int], config: FetchConfig
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}/{path.lstrip('/')}?{query}"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "ProtocolLens/0.1"},
        )
        for attempt in range(config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt == config.max_retries:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def iter_studies(self, config: FetchConfig) -> Iterator[dict[str, Any]]:
        """Yield studies across API pages up to the configured maximum."""
        returned = 0
        page_token: str | None = None
        while returned < config.max_studies:
            params: dict[str, str | int] = {
                "format": "json",
                "pageSize": min(config.page_size, config.max_studies - returned),
                "query.cond": config.condition,
                "filter.overallStatus": "|".join(config.statuses),
            }
            if page_token:
                params["pageToken"] = page_token
            payload = self._get_json("studies", params, config)
            studies = payload.get("studies", [])
            if not isinstance(studies, list):
                raise ValueError("API response field 'studies' is not a list")
            for study in studies:
                if returned >= config.max_studies:
                    return
                yield study
                returned += 1
            page_token = payload.get("nextPageToken")
            if not page_token or not studies:
                return


def normalize_study(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten one API study into a stable row for later NLP and modelling."""
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    conditions = protocol.get("conditionsModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    interventions = protocol.get("armsInterventionsModule", {})
    contacts = protocol.get("contactsLocationsModule", {})

    intervention_types = _unique(
        [item.get("type") for item in interventions.get("interventions", [])]
    )
    locations = contacts.get("locations", [])
    countries = _unique([item.get("country") for item in locations])
    lead_sponsor = sponsor.get("leadSponsor", {})
    enrollment = design.get("enrollmentInfo", {})
    design_info = design.get("designInfo", {})

    return {
        "nct_id": identification.get("nctId"),
        "brief_title": identification.get("briefTitle"),
        "official_title": identification.get("officialTitle"),
        "overall_status": status.get("overallStatus"),
        "why_stopped": status.get("whyStopped"),
        "study_type": design.get("studyType"),
        "phases": "|".join(design.get("phases", [])),
        "conditions": "|".join(conditions.get("conditions", [])),
        "intervention_types": "|".join(intervention_types),
        "lead_sponsor_name": lead_sponsor.get("name"),
        "lead_sponsor_class": lead_sponsor.get("class"),
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "allocation": design_info.get("allocation"),
        "intervention_model": design_info.get("interventionModel"),
        "primary_purpose": design_info.get("primaryPurpose"),
        "masking": _dig(design_info, "maskingInfo", "masking"),
        "minimum_age": eligibility.get("minimumAge"),
        "maximum_age": eligibility.get("maximumAge"),
        "sex": eligibility.get("sex"),
        "healthy_volunteers": eligibility.get("healthyVolunteers"),
        "eligibility_criteria": eligibility.get("eligibilityCriteria"),
        "start_date": _dig(status, "startDateStruct", "date"),
        "study_first_submit_date": status.get("studyFirstSubmitDate"),
        "study_first_submit_qc_date": status.get("studyFirstSubmitQcDate"),
        "study_first_post_date": _dig(status, "studyFirstPostDateStruct", "date"),
        "completion_date": _dig(status, "completionDateStruct", "date"),
        "last_update_post_date": _dig(status, "lastUpdatePostDateStruct", "date"),
        "number_of_locations": len(locations),
        "countries": "|".join(countries),
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
