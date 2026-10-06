# Data dictionary

## Identity and provenance

| Field | Meaning |
|---|---|
| `nct_id` | ClinicalTrials.gov study identifier and deduplication key |
| `brief_title`, `official_title` | Registry titles |
| `query_conditions` | Therapeutic-area query that collected the record; not a native registry field |
| `conditions` | Conditions listed in the registry record |

## Experimental target

| Field | Meaning |
|---|---|
| `overall_status` | Current registry status; only `COMPLETED` and `TERMINATED` enter modelling |
| `target_terminated` | Derived binary target: terminated = 1, completed = 0 |

`why_stopped` is retained for later error inspection but blocked from modelling.
The dashboard maps it to transparent descriptive categories only after filtering
to terminated studies; these categories never enter a feature transformer.

## Allowed structured model fields

Categorical: query area, study type, phase, intervention type, sponsor class,
allocation, intervention model, primary purpose, masking, sex, and healthy
volunteer status.

Numeric: parsed minimum/maximum age and submission year.

## Blocked leakage-prone fields

`overall_status`, `why_stopped`, completion date, last update date, enrollment
count/type, and current number of locations never enter a feature transformer.
Actual enrollment and current site counts can reflect events after trial launch.

## Eligibility NLP fields

| Field | Meaning |
|---|---|
| `eligibility_criteria` | Full free-text input for TF-IDF |
| `criterion_count` | Non-empty parsed criterion lines |
| `inclusion_count`, `exclusion_count` | Criteria assigned after section headers |
| `unclassified_count` | Lines not assigned to a detected section |
| `average_criterion_words` | Mean parsed criterion length |
| `numeric_constraint_count` | Numeric expressions in eligibility text |
| `comparator_count` | `<`, `>=`, “at least”, and related expressions |
| `temporal_constraint_count` | Hour/day/week/month/year mentions |
| `negation_count` | Transparent negation/exclusion keyword matches |
| `clinical_measure_count` | Matches to a documented clinical-measure dictionary |
| `complexity_index` | 0–100 capped descriptive formula; not a learned risk score |

## Dates

`validation_date` uses first submission date, then QC submission date, then start
date as fallbacks. It determines the chronological split and is not directly used
as an outcome-derived feature; only its year enters the structured allowlist.
