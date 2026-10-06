# Generated artifacts

This directory is recreated by the project scripts and is gitignored except for
this documentation.

- `retrieval/tfidf.joblib`: lexical retrieval index
- `retrieval/semantic/`: MiniLM embeddings, records, and configuration
- `retrieval/proxy_evaluation.json`: weak-label retrieval sanity check
- `retrieval/manual_evaluation.json`: final human metrics after all ratings
- `models/model_comparison.csv`: six holdout experiments
- `models/selection_comparison.csv`: middle-period model-selection results
- `models/best_uncalibrated.joblib`: selected experimental model
- `models/best_calibrated.joblib`: calibration experiment
- `models/training_summary.json`: split, metrics, intervals, and scope
- `models/global_feature_importance.csv`: global explanation table
- `models/subgroup_metrics.csv`: therapeutic-area performance
- `models/error_analysis.csv`: false-positive/false-negative cases
- `models/validation_comparison.json`: random-versus-chronological sensitivity

Do not commit model/data artifacts without first checking their size, source
terms, privacy, and reproducibility requirements.
