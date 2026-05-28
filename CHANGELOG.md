# Changelog

All notable changes to this project are documented here.

---

## [Unreleased]

### Added
- Full-page OCR with automatic line segmentation (`src/inference/predict_page.py`)
- Streamlit demo supporting both line-level and full-page OCR modes
- Hugging Face IAM dataset preparation workflow (`src/data/prepare_hf_iam.py`)
- Learning rate scheduler builder with `reduce_on_plateau` and `cosine` options
- CSV-based training metrics logger (`src/utils/logger.py`)
- Training curve visualization (`src/visualization/plot_metrics.py`)
- Test suite: charset, metrics, model, preprocessing, and decoder tests
- Helper scripts: `download_hf_iam.py`, `download_iam_kaggle.py`, `run_inference_demo.py`
- `pyproject.toml` with pytest and ruff configuration
- `Makefile` for common development tasks

### Changed
- Streamlit app updated to support full-page OCR mode

---

## [0.1.0] — Initial Release

### Added
- CRNN model with CNN feature extractor and BiLSTM sequence model
- CTC Loss training pipeline
- Greedy CTC decoder
- IAM Handwriting Dataset preparation from local files
- Character Error Rate (CER) and Word Error Rate (WER) evaluation
- Image preprocessing pipeline
- Config-driven training via `configs/config.yaml`
