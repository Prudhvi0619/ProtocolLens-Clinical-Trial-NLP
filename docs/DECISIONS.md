# Decision log

## 2026-08-31 — Original implementation

ProtocolLens is being built as a new project. The previously downloaded GitHub
repository may be studied for learning, but its code will not be copied or
represented as original work.

## 2026-08-31 — Primary target

The first supervised experiment uses `COMPLETED` and `TERMINATED` as transparent
registry-status labels. The project will explicitly state that this is a
classification task, not a clinical-success predictor.

## 2026-08-31 — Data source

Use the public ClinicalTrials.gov API v2 and save raw records, normalized data,
query parameters, timestamp, and class counts for reproducibility.

Completed and terminated records are requested separately within every selected
therapeutic area. Cross-area duplicates are removed by NCT identifier while the
matched query areas are retained.

## 2026-08-31 — No AI agent in the core scope

The core value comes from NLP, information retrieval, conventional machine
learning, calibration, and explainability. An agent or chatbot is not required.

## 2026-08-31 — Eligibility complexity

Eligibility complexity is a descriptive, rule-based profile. Its 0–100 index is
made from capped criterion length/count, numerical and temporal constraints,
comparators, clinical-measure mentions, and exclusion proportion. It is not a
learned risk score and every component is shown separately.

## 2026-09-01 — Leakage-conscious feature allowlist

Models use only predeclared structured fields, eligibility text, and transparent
text-derived counts. Current outcome, reason stopped, completion/update dates,
actual enrollment, and current location count are blocked. Because the API gives
the current record rather than the record as it existed at trial launch, the
experiment remains retrospective and cannot prove prospective performance.

## 2026-09-01 — Validation and probability scope

Trials are sorted by submission/start date: the oldest 60% train candidate
models, the next 20% select the model by average precision then ROC-AUC, and the
latest 20% remain the final test. The selected model is then refitted on the
oldest 80%. Sigmoid calibration uses cross-validation only inside that
development partition. Since collection balances the labels, the displayed
score must not be read as real-world termination prevalence.

## 2026-09-01 — Outcome-language sanitization

The eligibility-text audit found 244 exact matches for words such as
`completed`, `terminated`, and `stopped`. Most are legitimate eligibility
phrases, but every exact match is removed inside the NLP pipeline during both
fitting and inference to prevent ambiguous target-language leakage. The selected
model remains eligibility-NLP Logistic Regression after retraining.

## 2026-09-01 — Unique-pair human review

The retrieval sheet contains 60 method-level rows but only 54 unique
query/candidate pairs because six results are shared by TF-IDF and MiniLM. Each
intrinsic pair is judged once under blinding, and that rating is propagated to
both methods to avoid duplicated work and inconsistent relevance labels.
