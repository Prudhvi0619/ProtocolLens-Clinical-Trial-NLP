# Model card

## Intended use

ProtocolLens is an educational retrospective analysis of whether eligibility
text contains signal associated with a ClinicalTrials.gov study being recorded
as `COMPLETED` versus `TERMINATED`. It supports portfolio demonstration,
methodological learning, and exploratory analysis.

## Not intended for

- Predicting clinical efficacy or regulatory approval.
- Recommending eligibility criteria, treatments, or trial operations.
- Estimating real-world termination prevalence.
- Making decisions about patients, sites, sponsors, or investments.

## Data

The development cohort contains 1,794 unique studies from breast cancer, type 2
diabetes, and rheumatoid arthritis. Statuses were requested separately to form
an almost balanced experimental cohort. The oldest 60% trains candidate models,
the next 20% selects the model, and the latest 20% is held out for final testing.

## Features

Structured features use a strict allowlist. Eligibility NLP uses TF-IDF plus
transparent rule-based counts. Outcome, reason stopped, completion/update dates,
actual enrollment, and current location counts are explicitly excluded.
An automated lexical audit also checks eligibility text for explicit outcome
words such as `terminated`, `completed`, and `stopped`. The raw cohort contains
244 matches, commonly in legitimate phrases such as “completed prior therapy.”
Every exact match is removed inside the NLP pipeline before both fitting and
inference, and the post-sanitization audit contains zero matches. This does not
eliminate the broader current-record limitation described below.

## Current evidence

Eligibility-NLP Logistic Regression was selected on the middle chronological
partition (validation average precision 0.653), then refitted on the oldest 80%.
Its final latest-20% performance is:

- Average precision: 0.678 (bootstrap 95% CI 0.596–0.749)
- ROC-AUC: 0.726 (bootstrap 95% CI 0.669–0.778)
- Brier score: 0.216

The constant prior-probability baseline has average precision 0.429 and ROC-AUC
0.500 on the same final test, providing context for the learned-model results.

For Logistic Regression, NLP minus structured average precision is approximately
+0.117 (paired bootstrap 95% CI +0.036 to +0.199). This supports the value of
eligibility text within the tested linear model and retrospective cohort.

Sigmoid calibration worsened Brier score from 0.216 to 0.235 on the temporal
holdout. The project reports this negative result and does not market its scores
as population-calibrated probabilities.

Performance differs materially by therapeutic area: average precision is 0.793
for breast cancer, 0.742 for rheumatoid arthritis, and 0.481 for type 2 diabetes.
The overall result therefore must not be assumed to transfer uniformly across
conditions.

Top coefficients include generic registry wording and area-associated tokens
such as “trial,” “type II,” and “RA.” They can encode documentation style,
therapeutic-area mix, and temporal practice changes rather than genuine protocol
difficulty. Coefficient magnitude explains model arithmetic; it does not make a
clinical or causal feature claim.

A secondary random stratified split produced average precision 0.705 versus
0.678 on the chronological holdout. Other metrics did not move uniformly
(random ROC-AUC 0.705 versus chronological 0.726), reinforcing that the
chronological holdout—not whichever split yields the largest number—is primary.

The 54-pair blinded AI-assisted retrieval diagnostic produced Precision@5 0.933
for TF-IDF and 0.800 for semantic retrieval (NDCG@5 0.928 and 0.870,
respectively). These LLM-generated judgments are useful for qualitative error
analysis but are not an independent human gold standard.

## Limitations

The API exposes current records, not necessarily the version present at trial
launch. Eligibility text can be amended, so residual temporal leakage cannot be
ruled out. The cohort is not population-representative, three areas do not cover
all medicine, status data can be incomplete, and associations are not causal.
