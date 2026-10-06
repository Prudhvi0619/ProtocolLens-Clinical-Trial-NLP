"""Record hashes and software versions for the current generated evidence."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.provenance import file_sha256  # noqa: E402

EVIDENCE_FILES = [
    "data/raw/fetch_metadata.json",
    "data/raw/studies.jsonl",
    "data/processed/trials.csv",
    "data/processed/trials_with_features.csv",
    "artifacts/retrieval/metadata.json",
    "artifacts/retrieval/proxy_evaluation.json",
    "artifacts/retrieval/tfidf.joblib",
    "artifacts/retrieval/semantic/config.json",
    "artifacts/retrieval/semantic/embeddings.npy",
    "artifacts/models/model_comparison.csv",
    "artifacts/models/best_uncalibrated.joblib",
    "artifacts/models/training_summary.json",
    "artifacts/models/text_leakage_audit.json",
    "artifacts/models/validation_comparison.json",
]
PACKAGES = [
    "joblib",
    "numpy",
    "pandas",
    "plotly",
    "scikit-learn",
    "sentence-transformers",
    "streamlit",
]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def main() -> None:
    missing = [relative for relative in EVIDENCE_FILES if not (PROJECT_ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot build provenance manifest; missing: {missing}")
    files = {
        relative: {
            "sha256": file_sha256(PROJECT_ROOT / relative),
            "bytes": (PROJECT_ROOT / relative).stat().st_size,
        }
        for relative in EVIDENCE_FILES
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "files": files,
        "mutable_files_excluded": [
            "data/processed/retrieval_evaluation.csv",
            "data/processed/retrieval_evaluation_metadata.json",
            "artifacts/retrieval/manual_evaluation.json",
        ],
    }
    output_path = PROJECT_ROOT / "artifacts" / "run_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", **manifest}, indent=2))


if __name__ == "__main__":
    main()
