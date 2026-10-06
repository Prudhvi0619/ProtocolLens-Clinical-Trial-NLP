# Generated data

This directory intentionally tracks documentation and empty folders, not the
downloaded study records.

- `raw/studies.jsonl`: original ClinicalTrials.gov API study objects
- `raw/fetch_metadata.json`: query, timestamp, class counts, and output paths
- `processed/trials.csv`: normalized trial-level table
- `processed/trials_with_features.csv`: modelling table with eligibility features
- `processed/parsed_criteria.jsonl`: inclusion/exclusion parser output
- `processed/retrieval_evaluation.csv`: editable human-relevance judgments

Recreate these files using the commands in the project README. Clinical trial
records can change, so preserve fetch metadata when reporting results.
