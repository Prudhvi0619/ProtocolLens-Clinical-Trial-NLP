# Retrieval relevance evaluation guide

`data/processed/retrieval_evaluation.csv` contains one completed and one
terminated query from each therapeutic area, five results per method, and 60
method-level result rows. Six query/candidate pairs occur in both methods, so the
dashboard asks for 54 unique blinded comparisons and propagates those six shared
judgments to both method rows. This prevents inconsistent duplicate labels. The
dashboard hides the retrieval method, candidate outcome, and similarity score
while rating to reduce reviewer bias. Its focused review mode shows one pair at
a time, includes both eligibility texts, and records each rating with one click
before advancing to the next unrated pair.

For every result, compare the query and candidate title, therapeutic area,
phase, intervention type, and eligibility population. Enter:

- `0`: unrelated disease/population or not useful as a comparable trial.
- `1`: partly comparable, such as the same broad disease with a materially
  different population, phase, or intervention context.
- `2`: strongly comparable disease and reasonably similar population/design.

Add a short note for ambiguous judgments. Apply the same rubric to both methods
without favoring the method name. The dashboard saves the final metric artifact
automatically after judgment 60. You can also score the completed CSV directly:

```powershell
python scripts\score_retrieval_evaluation.py
```

## Current completed diagnostic

All 54 unique pairs are currently labeled by an AI-assisted rubric review, with
concise reasons stored in `reviewer_note` and `review_source=ai_assisted_llm`.
TF-IDF achieved Precision@5 0.933, MRR 1.000, and NDCG@5 0.928; semantic
retrieval achieved Precision@5 0.800, MRR 0.833, and NDCG@5 0.870. This reversal
from the shared-area proxy is useful error-analysis evidence, but the labels are
an LLM-based qualitative assessment and **not an independent human gold
standard**.

Report Precision@5, MRR, and NDCG@5 together with the review source. The
automatic therapeutic-area overlap evaluation is only a sanity check and must
never be presented as human relevance. The scoring command saves the final
evidence to the legacy internal path
`artifacts/retrieval/manual_evaluation.json`; the JSON provenance fields identify
the review correctly as AI-assisted. A future independent reviewer can reset the
sheet and repeat the same blinded workflow.
