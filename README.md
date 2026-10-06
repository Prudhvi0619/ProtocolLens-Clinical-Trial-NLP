# ProtocolLens

ProtocolLens is an interpretable clinical-trial eligibility intelligence system.
It parses eligibility criteria, retrieves comparable historical trials, and
tests whether eligibility NLP adds signal beyond structured registry fields for
`COMPLETED` versus `TERMINATED` classification.

This is a retrospective research demonstration—not a clinical-success model,
causal analysis, medical recommendation, or real-world risk calculator.

## Results snapshot

- 1,794 unique ClinicalTrials.gov studies across breast cancer, type 2 diabetes,
  and rheumatoid arthritis.
- Balanced experimental cohort: 900 completed and 894 terminated studies.
- Chronological design: 1,076 oldest studies for model fitting, 359 for model
  selection, then refitting on all 1,435 older studies before testing on 359
  newer studies beginning 21 August 2019.
- Eligibility-NLP Logistic Regression average precision: **0.678** (bootstrap
  95% CI 0.596–0.749) and ROC-AUC: **0.726** (95% CI 0.669–0.778).
- Logistic Regression NLP-minus-structured average precision: **+0.117**
  (paired-bootstrap 95% CI +0.036 to +0.199).
- Sigmoid calibration worsened temporal-holdout Brier score from 0.216 to 0.235;
  this negative result is shown rather than hidden.
- Semantic retrieval same-area Precision@5: 0.994 versus TF-IDF 0.933 on a
  300-query weak-label sanity check.
- A 54-pair blinded **AI-assisted diagnostic review** reversed that ordering:
  TF-IDF Precision@5 was **0.933** versus **0.800** for semantic retrieval
  (NDCG@5 0.928 versus 0.870). These LLM-generated judgments support error
  analysis but are not an independent human gold standard.

## What the application does

1. Downloads studies through ClinicalTrials.gov API v2 with pagination, retries,
   raw-response retention, and reproducibility metadata.
2. Parses inclusion and exclusion criteria and generates a transparent 0–100
   descriptive complexity profile.
3. Retrieves comparable trials using a TF-IDF baseline and MiniLM sentence
   embeddings.
4. Generates a structured blinded evaluation sheet and computes Precision@K,
   MRR, and NDCG after relevance review.
5. Compares structured-only, NLP-only, and combined Logistic Regression models
   against a prior-probability baseline on chronological data.
6. Reports discrimination, classification, calibration, confidence intervals,
   feature explanations, and known limitations.
7. Provides a tested Streamlit interface for cohort exploration, eligibility
   analysis, comparable-trial retrieval, and model evidence.

## Setup on Windows

For a one-time setup, double-click:

```text
setup_environment.bat
```

It creates an isolated `.venv` and installs the required packages. The equivalent
manual commands are:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10–3.12 is supported. Select `.venv\Scripts\python.exe` as the VS Code
interpreter; do not reuse a YOLO or unrelated project environment.

For development and linting, install `requirements-dev.txt` instead.

## Reproduce the project

Download a balanced multi-area development cohort:

```powershell
python scripts\fetch_trials.py `
  --condition "Breast Cancer" `
  --condition "Type 2 Diabetes" `
  --condition "Rheumatoid Arthritis" `
  --per-status 300
```

The fetch writes raw JSONL, a normalized CSV, and query metadata. Build the local
NLP, retrieval, evaluation, and model artifacts:

```powershell
python scripts\build_eligibility_features.py
python scripts\build_retrieval_indexes.py
python scripts\create_retrieval_evaluation.py
python scripts\evaluate_retrieval_proxy.py
python scripts\audit_text_leakage.py
python scripts\train_models.py
python scripts\compare_validation_strategies.py
python scripts\analyze_errors.py
python scripts\build_run_manifest.py
python scripts\validate_project.py
python scripts\validate_documentation.py
```

After the sentence model has been downloaded once, rebuild all local stages with:

```powershell
python scripts\run_local_pipeline.py
```

## Run the application

Double-click `run_dashboard.bat`, or run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none
```

Open `http://localhost:8501`. The eligibility page accepts new protocol text,
returns visible complexity components, retrieves semantic comparables, and shows
an explicitly uncalibrated retrospective model score.

The Retrieval review page provides a blinded, one-comparison-at-a-time workflow
with full eligibility context and automatically saves the final metrics after
all 60 result rows are labeled. Shared results are judged once, so the current
sheet requires 54 unique blinded comparisons, each saved with one click. The
dashboard visibly identifies the current labels as AI-assisted.

On Windows, you can also double-click `run_dashboard.bat` after setup.

## Retrieval relevance evaluation

Open `data/processed/retrieval_evaluation.csv` and label every candidate:

- `0`: irrelevant
- `1`: partly relevant
- `2`: highly relevant

Then run:

```powershell
python scripts\score_retrieval_evaluation.py
```

Use the rubric in [EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md). Do not present
the automatic same-area proxy or the current AI-assisted judgments as independent
human relevance. An independent blinded human review remains an optional stronger
validation step.

## Verification

```powershell
python -m unittest discover -s tests -v
python scripts\validate_project.py
python scripts\validate_documentation.py
```

The current suite has 31 tests, including every dashboard page, the complete
new-protocol analysis interaction, and safe manual-review persistence. The artifact validator checks cohort size,
class balance, ID uniqueness, feature bounds, retrieval alignment, all six model
variants, temporal split evidence, and explainability outputs.

GitHub Actions runs linting, formatting, and artifact-independent unit tests on
clean clones. The dashboard interaction test and artifact validator run locally
after the reproducible data/model pipeline has been built.

## Repository map

```text
app.py                          Streamlit application
src/protocollens/
  clinicaltrials.py             API client and normalized schema
  eligibility.py                Criteria parser and complexity features
  retrieval.py                  TF-IDF/semantic ranking and metrics
  modelling.py                  Feature allowlist, validation, models, metrics
scripts/                        Reproducible pipeline commands
tests/                          Unit and dashboard interaction tests
docs/                           Architecture, decisions, model card, guides
data/                           Generated raw and processed data (gitignored)
artifacts/                      Generated indexes and models (gitignored)
```

## Read before interpreting results

ClinicalTrials.gov exposes current records, which may differ from the record at
trial launch. A chronological split reduces but cannot eliminate temporal
leakage. Outcome and obvious post-outcome fields are blocked, the sample is
deliberately balanced, and all 244 raw exact outcome-word matches are stripped
inside the NLP pipeline before fitting and inference. Associations are not causes. See the full
[model card](docs/MODEL_CARD.md) and [decision log](docs/DECISIONS.md).

Additional documentation:

- [Architecture](docs/ARCHITECTURE.md)
- [Project state](docs/PROJECT_STATE.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Completion audit](docs/COMPLETION_AUDIT.md)
- [Interview learning guide](docs/LEARNING_GUIDE.md)
- [Résumé and interview material](docs/RESUME_AND_INTERVIEW.md)
