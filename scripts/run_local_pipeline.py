"""Rebuild all local artifacts after trials.csv has been downloaded."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["scripts/build_eligibility_features.py"],
    ["scripts/build_retrieval_indexes.py", "--local-files-only"],
    ["scripts/create_retrieval_evaluation.py"],
    ["scripts/evaluate_retrieval_proxy.py"],
    ["scripts/audit_text_leakage.py"],
    ["scripts/train_models.py"],
    ["scripts/compare_validation_strategies.py"],
    ["scripts/analyze_errors.py"],
    ["scripts/build_run_manifest.py"],
    ["scripts/validate_project.py"],
    ["scripts/validate_documentation.py"],
]


def main() -> None:
    for step in STEPS:
        command = [sys.executable, *step]
        print(f"\nRunning {' '.join(step)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("\nProtocolLens local pipeline completed successfully.")


if __name__ == "__main__":
    main()
