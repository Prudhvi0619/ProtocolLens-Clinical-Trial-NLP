"""Reliable helpers for blinded manual retrieval review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PAIR_COLUMNS = ["query_nct_id", "candidate_nct_id"]


def review_progress(frame: pd.DataFrame) -> tuple[int, int, int]:
    """Return labeled rows, completed unique pairs, and total unique pairs."""
    labelled_rows = int(frame["relevance_0_2"].notna().sum())
    pair_completion = frame.groupby(PAIR_COLUMNS, sort=False)["relevance_0_2"].apply(
        lambda values: values.notna().all()
    )
    return labelled_rows, int(pair_completion.sum()), len(pair_completion)


def apply_pair_judgment(
    frame: pd.DataFrame,
    query_nct_id: str,
    candidate_nct_id: str,
    relevance: int,
    reviewer_note: str = "",
) -> pd.DataFrame:
    """Apply one intrinsic relevance label to every method row for a pair."""
    if relevance not in {0, 1, 2}:
        raise ValueError("relevance must be 0, 1, or 2")
    updated = frame.copy()
    same_pair = (updated["query_nct_id"] == query_nct_id) & (
        updated["candidate_nct_id"] == candidate_nct_id
    )
    if not same_pair.any():
        raise KeyError(f"Unknown review pair: {query_nct_id}/{candidate_nct_id}")
    updated.loc[same_pair, "relevance_0_2"] = relevance
    updated.loc[same_pair, "reviewer_note"] = reviewer_note.strip()
    return updated


def save_review_progress(frame: pd.DataFrame, path: Path) -> None:
    """Atomically persist review progress so an interrupted write cannot erase labels."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)
