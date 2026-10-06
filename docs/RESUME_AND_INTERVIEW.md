# Résumé and interview material

## Project title

**ProtocolLens — NLP-Based Clinical-Trial Eligibility Intelligence**

## Résumé bullets

- Built an end-to-end NLP and ML pipeline over 1,794 ClinicalTrials.gov studies
  spanning three therapeutic areas, including API ingestion, inclusion/exclusion
  parsing, interpretable protocol-complexity features, and a tested Streamlit app.
- Implemented TF-IDF and MiniLM semantic comparable-trial retrieval with a
  60-result/54-unique-comparison blinded relevance workflow; semantic retrieval
  reached 0.994 same-area Precision@5 on a 300-query weak-label sanity check.
- Compared structured-only, NLP-only, and combined Logistic Regression models
  using separate chronological selection and final-test partitions;
  eligibility NLP achieved 0.678 average
  precision and a +0.117 paired-bootstrap improvement over structured Logistic
  Regression (95% CI +0.036 to +0.199).
- Added feature and lexical leakage controls, bootstrap confidence intervals, calibration analysis,
  subgroup/error analysis, feature explanations, automated tests, and explicit
  model limitations for reproducible and responsible evaluation.

All 54 unique AI-assisted comparisons have populated the 60 result rows. The
diagnostic favored TF-IDF (Precision@5 0.933) over semantic retrieval (0.800),
but do not present those numbers as independent human validation. The same-area
metric must always be labelled as a weak proxy. For the résumé, the conservative
bullet above emphasizes the implemented workflow; discuss the AI-assisted result
and its limitation when asked about evaluation.

## 90-second explanation

Clinical-trial eligibility criteria contain rich information about the target
population, but they are stored as unstructured text. ProtocolLens asks whether
that text adds useful signal beyond ordinary structured registry fields and also
uses it to find comparable historical trials. I collected 1,794 completed or
terminated studies through the ClinicalTrials.gov API across three therapeutic
areas. I built a deterministic parser for inclusion and exclusion criteria, an
interpretable complexity profile, a TF-IDF baseline, and MiniLM semantic
retrieval. For the modelling experiment I compared structured-only, NLP-only,
and combined Logistic Regression models using the oldest 60 percent for fitting,
the next 20 percent for selection, and the latest 20 percent for
final testing. Eligibility NLP improved Logistic Regression average precision from
0.557 to 0.678, with a positive paired-bootstrap interval. However, the combined
model did not win, and sigmoid calibration became worse on the temporal holdout.
I expose those negative results, leakage limitations, retrieval evaluation, and
feature explanations in a Streamlit dashboard. The output is a retrospective
registry-status signal, not clinical-success or medical advice.

For lexical leakage control, the model pipeline removes exact registry-outcome
words at both fitting and inference. The audit found 244 raw mentions and zero
remaining after sanitization; performance changed only slightly after retraining.

## Five-minute demo order

1. Overview: show cohort balance and the complexity distribution.
2. Eligibility analyzer: paste criteria and explain visible parser outputs.
3. Click the evidence analysis button: show comparable trials and carefully
   describe the uncalibrated score.
4. Comparable trials: contrast exact-term TF-IDF with semantic retrieval.
5. Model evidence: compare all six variants, the bootstrap NLP improvement, and
   the failed calibration result.
6. Limitations: finish by explaining current-record leakage and label scope.

## Questions to expect

- Why is average precision useful alongside ROC-AUC?
- Why can balancing the labels invalidate a population probability claim?
- Which blocked columns would have leaked outcome information?
- Why did you use a temporal holdout instead of only random cross-validation?
- Why can current ClinicalTrials.gov eligibility text still leak future edits?
- Why might TF-IDF outperform embeddings for some exact clinical terminology?
- Why did combined features underperform NLP alone?
- What does a negative calibration result tell you about temporal drift?
- How would you obtain a truly prospective dataset and external validation set?
- What would you change before any operational or clinical use?
