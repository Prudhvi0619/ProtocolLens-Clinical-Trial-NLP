# Completion audit

| Requirement | Status | Authoritative evidence |
|---|---|---|
| ClinicalTrials.gov API pipeline | Complete | `clinicaltrials.py`, `fetch_trials.py`, raw metadata, pagination test |
| Multiple therapeutic areas | Complete | 1,794-row dataset; three query areas; artifact validator |
| Eligibility parser | Complete | `eligibility.py`, parsed JSONL, parser tests |
| Transparent complexity profile | Complete | visible formula/components, 0–100 bound test, dashboard |
| TF-IDF retrieval baseline | Complete | persisted 50,000-term index and ranking tests |
| Sentence-transformer retrieval | Complete | MiniLM 384-dimensional index and semantic ranking test |
| Retrieval explanations | Complete | shared weighted terms; dashboard interaction test |
| Weak-label retrieval sanity check | Complete | 300-query `proxy_evaluation.json` with explicit warning |
| Retrieval evaluation workflow | Complete | 60 result rows, 54 unique blinded comparisons, full protocol context, atomic progress saving |
| AI-assisted retrieval judgments | Complete | 60/60 method rows labeled with concise reasons and explicit `ai_assisted_llm` provenance |
| Independent human retrieval review | Optional future validation | Current judgments are not an independent human gold standard |
| Structured-only model | Complete | Logistic Regression final-test row |
| NLP-only model | Complete | Logistic Regression final-test row |
| Combined model | Complete | Logistic Regression final-test row |
| Naïve comparison baseline | Complete | constant-prior final-test row |
| Temporal selection and test | Complete | oldest 60% fit, middle 20% select, latest 20% final test |
| Leakage controls | Complete | feature allowlist/blocklist, 244 raw exact-word matches removed at fit/inference, zero post-sanitization matches |
| Confidence intervals | Complete | 500-sample bootstrap intervals and paired NLP comparison |
| Calibration | Complete | sigmoid CV on development data; negative temporal result retained |
| Explainability | Complete | global coefficients and signed local contributions |
| Error/subgroup analysis | Complete | error and therapeutic-area CSV artifacts |
| Random-vs-temporal sensitivity | Complete | `validation_comparison.json` |
| Streamlit dashboard | Complete | six pages plus new-protocol interaction test and live health check |
| Automated quality | Complete | Ruff lint/format and 31 passing tests |
| Artifact provenance | Complete | SHA-256 and package-version run manifest; hashes checked by validator |
| End-to-end rebuild | Complete | `run_local_pipeline.py` completed and artifact validator passed |
| README, architecture, model card | Complete | repository documentation |
| Interview and résumé material | Complete | learning and résumé guides |

## Final status

The technical implementation and AI-assisted diagnostic review are complete.
TF-IDF achieved Precision@5 0.933, MRR 1.000, and NDCG@5 0.928; semantic
retrieval achieved Precision@5 0.800, MRR 0.833, and NDCG@5 0.870. These are
supporting qualitative results, not independent human validation. To verify:

```powershell
python scripts\validate_project.py
```

The standalone `score_retrieval_evaluation.py` command remains available if the
CSV is relabeled outside the dashboard. Any quoted retrieval metric must include
its review source; do not call the current results human evaluation.
