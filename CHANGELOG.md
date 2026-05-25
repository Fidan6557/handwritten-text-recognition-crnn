# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- `src/training/scheduler.py` — ReduceLROnPlateau and CosineAnnealing scheduler builder
- `src/utils/logger.py` — CSV-based training metrics logger
- `src/visualization/plot_metrics.py` — Training curves and prediction sample plots
- `tests/test_metrics.py` — CER and WER unit tests
- `tests/test_preprocessing.py` — Image preprocessing unit tests
- `tests/test_model.py` — CRNN architecture unit tests
- `tests/test_charset.py` — Character vocabulary unit tests
- `scripts/download_iam_kaggle.py` — IAM dataset download via Kaggle CLI
- `scripts/download_hf_iam.py` — IAM dataset download via Hugging Face Hub
- `scripts/run_inference_demo.py` — Quick inference demo script
- `Makefile` — Developer shortcuts for install, test, train, demo
- `pyproject.toml` — Pytest and Ruff configuration
- `CONTRIBUTING.md` — Contribution guidelines

## [0.1.0] — Initial Release

### Added
- CRNN model with CNN + BiLSTM + CTC Loss
- Greedy CTC decoder
- IAM line dataset loader
- Image preprocessing pipeline
- Full-page segmentation with contour detection
- Training loop with validation
- CER and WER evaluation metrics
- Streamlit demo application
- IAM dataset preparation scripts (local and Hugging Face)
