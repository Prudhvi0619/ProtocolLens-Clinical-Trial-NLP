"""Comparable-trial retrieval using lexical and semantic representations."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def compose_trial_text(record: dict[str, Any]) -> str:
    """Create the searchable text while retaining field meaning."""
    parts = [
        f"Title: {record.get('brief_title', '')}",
        f"Conditions: {record.get('conditions', '')}",
        f"Phase: {record.get('phases', '')}",
        f"Study type: {record.get('study_type', '')}",
        f"Interventions: {record.get('intervention_types', '')}",
        f"Eligibility: {record.get('eligibility_criteria', '')}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


def _result(record: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "similarity": round(float(score), 6),
        "nct_id": record.get("nct_id"),
        "brief_title": record.get("brief_title"),
        "overall_status": record.get("overall_status"),
        "query_conditions": record.get("query_conditions"),
        "conditions": record.get("conditions"),
        "phases": record.get("phases"),
        "eligibility_criteria": record.get("eligibility_criteria"),
    }


def rank_scores(
    scores: Sequence[float],
    records: Sequence[dict[str, Any]],
    k: int = 5,
    exclude_nct_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert similarity values to stable ranked result dictionaries."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(scores) != len(records):
        raise ValueError("scores and records must have equal length")
    ordered = np.argsort(-np.asarray(scores), kind="stable")
    results: list[dict[str, Any]] = []
    for index in ordered:
        record = records[int(index)]
        if exclude_nct_id and record.get("nct_id") == exclude_nct_id:
            continue
        results.append(_result(record, len(results) + 1, float(scores[int(index)])))
        if len(results) == k:
            break
    return results


class TfidfTrialRetriever:
    """Lexical retriever backed by one shared sparse TF-IDF index."""

    method_name = "tfidf"

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("records must not be empty")
        self.documents = [compose_trial_text(record) for record in self.records]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50_000,
            sublinear_tf=True,
        )
        # Fit one vocabulary across all indexed trials; each row represents one trial.
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self._id_to_index = {
            record.get("nct_id"): index for index, record in enumerate(self.records)
        }

    def search(
        self, query_text: str, k: int = 5, exclude_nct_id: str | None = None
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        # Reuse the index vocabulary so query and trial vectors share columns.
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        return rank_scores(scores, self.records, k, exclude_nct_id)

    def search_by_id(self, nct_id: str, k: int = 5) -> list[dict[str, Any]]:
        if nct_id not in self._id_to_index:
            raise KeyError(f"Unknown NCT identifier: {nct_id}")
        index = self._id_to_index[nct_id]
        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        return rank_scores(scores, self.records, k, nct_id)

    def _overlap_terms(self, query_vector, candidate_index: int, top_n: int) -> list[dict]:
        overlap = query_vector.multiply(self.matrix[candidate_index]).tocoo()
        if overlap.nnz == 0:
            return []
        names = self.vectorizer.get_feature_names_out()
        ordered = np.argsort(-overlap.data, kind="stable")[:top_n]
        return [
            {
                "term": str(names[int(overlap.col[position])]),
                "overlap_weight": round(float(overlap.data[position]), 6),
            }
            for position in ordered
        ]

    def explain_pair(self, query_nct_id: str, candidate_nct_id: str, top_n: int = 10) -> list[dict]:
        """Top weighted TF-IDF terms shared by two indexed trials."""
        if query_nct_id not in self._id_to_index or candidate_nct_id not in self._id_to_index:
            raise KeyError("Both query and candidate NCT identifiers must be indexed")
        query_index = self._id_to_index[query_nct_id]
        candidate_index = self._id_to_index[candidate_nct_id]
        return self._overlap_terms(self.matrix[query_index], candidate_index, top_n)

    def explain_text_candidate(
        self, query_text: str, candidate_nct_id: str, top_n: int = 10
    ) -> list[dict]:
        """Top weighted TF-IDF terms shared by new text and an indexed trial."""
        if candidate_nct_id not in self._id_to_index:
            raise KeyError(f"Unknown NCT identifier: {candidate_nct_id}")
        query_vector = self.vectorizer.transform([query_text])
        return self._overlap_terms(query_vector, self._id_to_index[candidate_nct_id], top_n)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> TfidfTrialRetriever:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a TF-IDF retriever")
        return loaded


class SemanticTrialRetriever:
    """Semantic retriever backed by normalized 384-dimensional MiniLM vectors."""

    method_name = "sentence_transformer"

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        model_name: str = DEFAULT_SENTENCE_MODEL,
        embeddings: np.ndarray | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("records must not be empty")
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.documents = [compose_trial_text(record) for record in self.records]
        self._model = None
        if embeddings is None:
            embeddings = self._encode(self.documents)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(self.embeddings) != len(self.records):
            raise ValueError("embedding count must equal record count")
        self._id_to_index = {
            record.get("nct_id"): index for index, record in enumerate(self.records)
        }

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
                local_files_only=self.local_files_only,
            )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        # Normalization makes the later dot product equivalent to cosine similarity.
        return self._get_model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )

    def search(
        self, query_text: str, k: int = 5, exclude_nct_id: str | None = None
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        query_embedding = self._encode([query_text])[0]
        scores = self.embeddings @ query_embedding
        return rank_scores(scores, self.records, k, exclude_nct_id)

    def search_by_id(self, nct_id: str, k: int = 5) -> list[dict[str, Any]]:
        if nct_id not in self._id_to_index:
            raise KeyError(f"Unknown NCT identifier: {nct_id}")
        index = self._id_to_index[nct_id]
        scores = self.embeddings @ self.embeddings[index]
        return rank_scores(scores, self.records, k, nct_id)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "records.json").write_text(
            json.dumps(self.records, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "local_files_only": self.local_files_only,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> SemanticTrialRetriever:
        records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        embeddings = np.load(directory / "embeddings.npy")
        return cls(
            records,
            config["model_name"],
            embeddings=embeddings,
            # Building embeddings proves the model was already downloaded on
            # this machine. Avoid slow network metadata checks during dashboard use.
            local_files_only=True,
        )


def precision_at_k(relevances: Sequence[int], k: int, threshold: int = 1) -> float:
    selected = list(relevances[:k])
    if not selected:
        return 0.0
    return sum(value >= threshold for value in selected) / len(selected)


def reciprocal_rank(relevances: Sequence[int], threshold: int = 1) -> float:
    for rank, value in enumerate(relevances, start=1):
        if value >= threshold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    selected = np.asarray(relevances[:k], dtype=float)
    if selected.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, selected.size + 2))
    dcg = np.sum((np.power(2.0, selected) - 1.0) / discounts)
    ideal = np.sort(selected)[::-1]
    ideal_dcg = np.sum((np.power(2.0, ideal) - 1.0) / discounts)
    return float(dcg / ideal_dcg) if ideal_dcg else 0.0


def summarize_judgments(
    rows: Iterable[Mapping[str, Any]], k: int = 5
) -> dict[str, dict[str, float]]:
    """Aggregate completed 0–2 manual judgments by retrieval method."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get("relevance_0_2")
        if value is None or str(value).strip() in {"", "nan"}:
            raise ValueError("All manual relevance judgments must be completed")
        groups[(str(row["query_nct_id"]), str(row["method"]))].append(row)
    method_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_, method), group in groups.items():
        ordered = sorted(group, key=lambda row: int(row["rank"]))
        values = [int(float(row["relevance_0_2"])) for row in ordered]
        method_scores[method]["precision_at_k"].append(precision_at_k(values, k))
        method_scores[method]["mrr"].append(reciprocal_rank(values))
        method_scores[method]["ndcg_at_k"].append(ndcg_at_k(values, k))
    return {
        method: {metric: round(float(np.mean(values)), 4) for metric, values in metrics.items()}
        for method, metrics in method_scores.items()
    }


def select_review_query_ids(records: Sequence[Mapping[str, Any]], per_area: int = 2) -> list[str]:
    """Select deterministic, status-balanced review queries within each area."""
    if per_area < 1:
        raise ValueError("per_area must be at least 1")
    area_order = list(dict.fromkeys(str(record.get("query_conditions")) for record in records))
    selected: list[str] = []
    for area in area_order:
        area_records = [record for record in records if str(record.get("query_conditions")) == area]
        area_selected: list[str] = []
        for status in ("COMPLETED", "TERMINATED"):
            match = next(
                (
                    record
                    for record in area_records
                    if record.get("overall_status") == status
                    and record.get("nct_id") not in area_selected
                ),
                None,
            )
            if match and len(area_selected) < per_area:
                area_selected.append(str(match["nct_id"]))
        for record in area_records:
            nct_id = str(record.get("nct_id"))
            if len(area_selected) == per_area:
                break
            if nct_id not in area_selected:
                area_selected.append(nct_id)
        selected.extend(area_selected)
    return selected
