# Project state

## Frozen objective

Build an interpretable system that uses ClinicalTrials.gov data to parse trial
eligibility criteria, retrieve comparable historical studies, and evaluate
whether eligibility-text features improve completed-versus-terminated trial
classification beyond structured fields.

## Completed

- Reproducible API ingestion with retries and pagination.
- Separate sampling by status across three therapeutic areas.
- 1,794 unique trials with raw JSONL, normalized CSV, and query metadata.
- Rule-based inclusion/exclusion parsing.
- Transparent eligibility complexity features and parsed-criteria output.
- TF-IDF and MiniLM semantic retrieval indexes.
- Weak-label retrieval sanity check and a 60-row/54-unique-pair blinded review sheet.
- Completed AI-assisted relevance diagnostic with per-pair reasons and explicit
  non-human provenance.
- Three structured/NLP/combined Logistic Regression experiments.
- Chronological holdout, bootstrap intervals, calibration, and explanations.
- Therapeutic-area subgroup metrics and high-confidence error analysis.
- Six-page Streamlit dashboard with new-protocol analysis and retrieval review.
- 31 passing tests plus passing artifact, provenance, and documentation-consistency gates.

## Current milestone

The implementation, evidence generation, AI-assisted retrieval diagnostic, and
automated verification are complete. The next milestone is learning the system,
rehearsing the interview explanation, and optionally obtaining an independent
human relevance review.

## Guardrails

- Do not describe model output as a causal explanation of trial termination.
- Do not present model output as medical advice or an operational decision.
- Prevent future information leakage by keeping post-outcome fields out of model
  features.
- Prefer temporal validation to a random-only split when dates support it.
- Report class balance, calibration, and error analysis, not accuracy alone.
- Never describe the current AI-assisted retrieval judgments as independent
  human validation.
