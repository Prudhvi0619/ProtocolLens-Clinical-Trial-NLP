# Interview learning guide

This guide will grow with each milestone. For every component, be able to answer:

1. What user or analytical problem does it solve?
2. What data enters and what output leaves?
3. Why was this method selected over a simpler alternative?
4. How was it evaluated?
5. What can fail, leak, or be misinterpreted?

## Milestone 1 talking points

- ClinicalTrials.gov contains both structured fields and free-text eligibility.
- Raw API records are retained so processing is auditable and repeatable.
- A normalized table makes later exploration and modelling consistent.
- The initial labels are registry statuses, not proof of clinical efficacy.

## One descriptive outcome insight

The transparent reason mapper assigns 281 of 894 terminated trials to
recruitment/enrollment, the largest named category in this cohort. This is a
post-outcome descriptive observation, not a predictive feature or causal claim;
the large “other stated reason” group also shows the limits of simple rules.

## Explain the project in 45 seconds

ProtocolLens asks whether free-text clinical-trial eligibility criteria add
useful information beyond structured registry fields. I built an API pipeline
for 1,794 trials across three therapeutic areas, parsed inclusion and exclusion
criteria into an interpretable complexity profile, and implemented both TF-IDF
and sentence-transformer retrieval for comparable historical trials. I compared
structured-only, NLP-only, and combined Logistic Regression models on a
chronological holdout. Eligibility NLP improved Logistic Regression average
precision, but combined features were not automatically best and calibration
worsened on newer trials. I expose those results and limitations in a tested
Streamlit dashboard instead of presenting the model as a clinical decision tool.

## Questions you must be ready for

### Why use both TF-IDF and sentence transformers?

TF-IDF is fast, transparent, and strong when exact clinical terms overlap.
Sentence embeddings can retrieve semantically similar wording even without exact
token overlap. A blinded relevance sheet compares them rather than assuming the
larger model is better. In the completed 54-pair AI-assisted diagnostic, TF-IDF
Precision@5 was 0.933 versus 0.800 for semantic retrieval. That result is useful
for error analysis, but it is not an independent human gold standard.

### Why use a chronological split?

A random split allows older and newer studies to mix, which can overstate how a
model generalizes to future records. ProtocolLens uses the oldest 60% for initial
fitting, the next 20% for selection, and the latest 20% for final testing. The
latest partition is harder and more realistic, though current-record API
snapshots still prevent a true prospective claim.

### Why report average precision, ROC-AUC, and a baseline?

The positive class is termination, so precision-recall metrics focus on ranking
that class. Average precision should be compared with positive-class prevalence:
the prior baseline is about 0.429 while the selected model reaches 0.678.
ROC-AUC adds a threshold-independent ranking view.

The baseline shows why F1 alone can mislead: it obtains roughly 0.60 F1 but 0.50
ROC-AUC because its constant score crosses the default threshold and labels
nearly everything positive. Balanced accuracy, probability metrics, and the
confusion pattern reveal that it has no ranking skill.

### Did NLP improve the model?

Yes on this holdout: average precision rose from 0.557 with structured features
to 0.678 with eligibility NLP. The paired bootstrap interval for the improvement
was positive. The combined representation did not improve over NLP-only, so more
features did not automatically produce a better model.

### Why did the combined model not win?

More features do not guarantee better generalization. Structured variables may
add noise, interact poorly with sparse text, or drift over time. This is why the
project treats feature combinations as experiments rather than assumptions.

### Are the strongest words clinically important?

Not necessarily. Some top coefficients are generic wording or disease markers,
which may capture documentation style, therapeutic-area composition, or time.
They explain what the classifier used but do not establish clinical importance
or causality. This is also consistent with the weak type 2 diabetes subgroup.

### Why not call the output trial success probability?

The target is registry status, the sample deliberately balances classes, and
records may be amended. Calling it clinical success or real-world risk would be
scientifically misleading.

### Where is RAG or an AI agent?

Neither is required. Comparable-trial retrieval is an information-retrieval
component, not generation. The core problem is better served by auditable NLP,
retrieval evaluation, conventional ML, and explainability.
