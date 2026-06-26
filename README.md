# Handwritten Text Recognition with CRNN, BiLSTM and CTC Loss

[![CI](https://github.com/Fidan6557/handwritten-text-recognition-crnn/actions/workflows/ci.yml/badge.svg)](https://github.com/Fidan6557/handwritten-text-recognition-crnn/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end handwritten text recognition project built with PyTorch.  
The system recognizes handwritten text line images and converts them into machine-readable text using a CRNN architecture with CNN feature extraction, BiLSTM sequence modeling, and CTC Loss.

---

## Project Overview

Handwritten text recognition is a challenging computer vision task because handwriting varies by writer, style, spacing, stroke thickness, and image quality.

This project focuses on line-level handwritten text recognition using the IAM Handwriting Dataset. Instead of treating OCR as simple image classification, the model learns to predict a sequence of characters from an input image.

The project demonstrates:

- Computer vision preprocessing
- CNN-based feature extraction
- Sequence modeling with BiLSTM
- Alignment-free training with CTC Loss
- CTC decoding
- OCR evaluation using CER and WER
- Full-page line segmentation
- Streamlit-based demo interface

---

## Model Architecture

```text
Input Handwritten Text Image
        ↓
Image Preprocessing
        ↓
CNN Feature Extractor
        ↓
Feature Map to Sequence
        ↓
BiLSTM Sequence Model
        ↓
Linear Character Classifier
        ↓
CTC Decoder
        ↓
Predicted Text
```

---

## Key Features

- Line-level handwritten text recognition
- CRNN architecture implemented in PyTorch
- CNN feature extractor for visual representation
- BiLSTM layers for sequence understanding
- CTC Loss for alignment-free OCR training
- Greedy CTC decoder
- Character Error Rate (CER) and Word Error Rate (WER) evaluation
- Learning rate scheduling (ReduceLROnPlateau or CosineAnnealing)
- CSV-based training metrics logger
- Training curve and prediction sample visualization
- Full-page OCR with automatic line segmentation
- Streamlit application for image upload and inference
- Modular, config-driven, and reproducible project structure

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| OpenCV | Image preprocessing and segmentation |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Streamlit | Demo application |
| Matplotlib | Visualization |
| PyYAML | Config file parsing |
| editdistance | CER / WER metric computation |

---

## Project Structure

```text
handwritten-text-recognition-crnn/
│
├── app/
│   └── streamlit_app.py             # Streamlit demo interface
│
├── configs/
│   └── config.yaml                  # Centralized project configuration
│
├── data/
│   ├── raw/iam/                     # Raw IAM dataset (not tracked by Git)
│   └── processed/                   # Preprocessed train/val/test CSVs
│
├── outputs/
│   ├── checkpoints/                 # Saved model weights (not tracked by Git)
│   ├── logs/                        # Training CSV logs (not tracked by Git)
│   └── plots/                       # Training curve plots
│
├── sample_images/
│   └── page_sample.jpg              # Example input image
│
├── scripts/
│   ├── download_hf_iam.py           # Download IAM via Hugging Face Hub
│   ├── download_iam_kaggle.py       # Download IAM via Kaggle CLI
│   └── run_inference_demo.py        # Quick CLI inference demo
│
├── src/
│   ├── data/
│   │   ├── dataset.py               # PyTorch Dataset and collate_fn
│   │   ├── prepare_hf_iam.py        # Prepare IAM data from Hugging Face
│   │   ├── prepare_iam.py           # Prepare IAM data from local files
│   │   ├── preprocessing.py         # Image preprocessing pipeline
│   │   └── segmentation.py          # Full-page line segmentation
│   │
│   ├── inference/
│   │   ├── decoder.py               # Greedy CTC decoder
│   │   ├── predict.py               # Single-line OCR inference
│   │   └── predict_page.py          # Full-page OCR inference
│   │
│   ├── models/
│   │   └── crnn.py                  # CRNN model definition
│   │
│   ├── training/
│   │   ├── evaluate.py              # Validation loop with CER/WER
│   │   ├── scheduler.py             # LR scheduler builder
│   │   └── train.py                 # Main training script
│   │
│   ├── utils/
│   │   ├── charset.py               # Character vocabulary builder
│   │   ├── logger.py                # CSV metrics logger
│   │   ├── metrics.py               # CER and WER functions
│   │   └── seed.py                  # Reproducibility seed setter
│   │
│   └── visualization/
│       └── plot_metrics.py          # Training curve and prediction plots
│
├── tests/
│   ├── test_charset.py
│   ├── test_decoder.py
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Dataset

This project is designed for the [IAM Handwriting Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

### Option 1 — Download via Hugging Face Hub

```bash
make prepare-hf-iam
```

### Option 2 — Download via Kaggle CLI

```bash
make prepare-iam
```

The expected CSV format after preprocessing:

```csv
image_path,text
data/raw/iam/lines/a01/a01-000u/a01-000u-00.png,A MOVE to stop Mr. Gaitskell from
data/raw/iam/lines/a01/a01-000u/a01-000u-01.png,nominating any more Labour life Peers
```

The model focuses on line-level OCR. Full-page OCR with automatic line segmentation is also supported.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Fidan6557/handwritten-text-recognition-crnn.git
cd handwritten-text-recognition-crnn
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### Quick start with defaults

```bash
make train
```

Training will:
1. Read all hyperparameters from `configs/config.yaml`
2. Build the character vocabulary and save it to `outputs/charset.json`
3. Log metrics per epoch to `outputs/logs/train_<timestamp>.csv`
4. Save the best checkpoint to `outputs/checkpoints/best_crnn.pth`

---

## Evaluation

Evaluation runs automatically after each training epoch and reports:

| Metric | Meaning |
|---|---|
| CER | Character Error Rate — lower is better |
| WER | Word Error Rate — lower is better |

To plot training curves after training:

```bash
make plot-metrics
```

---

## Inference

### Single line image

```bash
make predict-line IMAGE=path/to/line.png
# or
python -m src.inference.predict --image path/to/line.png
```

### Full page image

```bash
make predict-page
# or
python -m src.inference.predict_page --image sample_images/page_sample.jpg --save-lines
```

### Quick CLI demo

```bash
python scripts/run_inference_demo.py --image sample_images/page_sample.jpg
```

---

## Run Streamlit Demo

```bash
make demo
```

The demo supports both line-level OCR and full-page OCR with automatic line segmentation.  
A trained checkpoint at `outputs/checkpoints/best_crnn.pth` is required to run inference.

---

## Training Pipeline

```text
1. Prepare IAM line-level dataset
2. Generate train / validation / test CSV files
3. Train CRNN model with CTC Loss
4. Monitor CER and WER per epoch
5. Save best model checkpoint
6. Use checkpoint in Streamlit demo or inference scripts
```

---

## Configuration

All training parameters are controlled from `configs/config.yaml`:

```yaml
training:
  batch_size: 16
  epochs: 30
  learning_rate: 0.0003
  weight_decay: 0.00001
  scheduler: reduce_on_plateau   # reduce_on_plateau | cosine
```

---

## Development

Run tests:

```bash
make test
```

Lint code:

```bash
make lint
```

Clean Python cache:

```bash
make clean
```

---

## Future Improvements

- Add beam search decoding
- Add data augmentation pipeline
- Add experiment tracking (MLflow or Weights & Biases)
- Add model training results and benchmarks
- Deploy the Streamlit demo online (Streamlit Cloud or HuggingFace Spaces)
- Compare CRNN with transformer-based OCR models (TrOCR)

---

## Author

Developed by [Fidan6557](https://github.com/Fidan6557)

---

## License

This project is licensed under the [MIT License](LICENSE).
