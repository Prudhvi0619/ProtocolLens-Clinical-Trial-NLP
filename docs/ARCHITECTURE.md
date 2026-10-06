# ProtocolLens architecture

```text
ClinicalTrials.gov API v2
          |
          v
 raw studies.jsonl + fetch_metadata.json
          |
          v
 normalized trials.csv
          |
          +-----------------------------+
          |                             |
          v                             v
 eligibility parser              structured allowlist
 + complexity profile            (leakage fields blocked)
          |                             |
          +--------------+--------------+
                         |
             +-----------+-----------+
             |                       |
             v                       v
     comparable-trial search     model experiment
     - TF-IDF                    - structured only
     - MiniLM embeddings         - NLP only
     - blinded relevance review  - combined
             |                   - chronological test
             |                   - lexical sanitization
             |                   - calibration/explanation
             +-----------+-----------+
                         |
                         v
                  Streamlit dashboard
```

## Component boundaries

- `clinicaltrials.py` knows the external API and normalized data contract.
- `eligibility.py` contains deterministic parsing and descriptive features.
- `retrieval.py` owns ranking, persistence, and retrieval metrics.
- Retrieval explanations expose shared weighted TF-IDF terms. For semantic
  results these are explicitly labelled post-hoc lexical evidence, not an
  explanation of the MiniLM embedding internals.
- `modelling.py` owns the feature allowlist, exact outcome-language sanitizer,
  temporal split, estimators, metrics, calibration tables, bootstrap intervals,
  and global explanations.
- `review.py` propagates shared-pair judgments and atomically saves review
  progress so an interrupted write cannot erase prior labels. Review provenance
  distinguishes AI-assisted labels from independent human judgments.
- New-protocol scores expose signed Logistic Regression contributions for the
  current row; these explain model arithmetic, not causal mechanisms.
- `app.py` reads generated artifacts and only mutates the retrieval-review CSV and
  its completed metric artifact; it never retrains models.

## Reproducibility

Raw API responses, normalized data, query metadata, fitted artifacts, evaluation
tables, and training summaries are separate. `run_manifest.json` records SHA-256
hashes for 14 immutable evidence files plus Python/package versions; the artifact
validator recomputes every hash. Generated data and models are not committed by
default because they are reproducible and can be large.
