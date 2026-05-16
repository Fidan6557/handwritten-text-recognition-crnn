# Handwritten Text Recognition with CRNN, BiLSTM and CTC Loss

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
- Character Error Rate evaluation
- Word Error Rate evaluation
- Streamlit application for image upload and preview
- Modular and reproducible project structure

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| OpenCV | Image preprocessing |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Streamlit | Demo application |
| Matplotlib | Visualization |
| CTC Loss | Sequence alignment-free training |

---

## Project Structure

```text
handwritten-text-recognition-crnn/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── prepare_iam.py
│   │
│   ├── inference/
│   │   ├── decoder.py
│   │   └── predict.py
│   │
│   ├── models/
│   │   └── crnn.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── charset.py
│       ├── metrics.py
│       └── seed.py
│
├── tests/
│   └── test_decoder.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project is designed for the IAM Handwriting Dataset.

The recommended format after preprocessing is:

```csv
image_path,text
data/raw/iam/lines/a01/a01-000u/a01-000u-00.png,A MOVE to stop Mr. Gaitskell from
data/raw/iam/lines/a01/a01-000u/a01-000u-01.png,nominating any more Labour life Peers
```

The model currently focuses on line-level OCR. Full-page OCR and automatic line segmentation can be added as future improvements.

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
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

The current demo supports image upload and preview.  
Full OCR prediction becomes available after training the model and saving the checkpoint.

---

## Training Pipeline

Planned training workflow:

```text
1. Prepare IAM line-level dataset
2. Generate train / validation / test CSV files
3. Build character vocabulary
4. Train CRNN model with CTC Loss
5. Evaluate using CER and WER
6. Save best model checkpoint
7. Use checkpoint in Streamlit demo
```

Expected checkpoint path:

```text
outputs/checkpoints/best_crnn.pth
```

Expected charset path:

```text
outputs/charset.json
```

---

## Evaluation Metrics

This project uses OCR-specific evaluation metrics.

| Metric | Meaning |
|---|---|
| CER | Character Error Rate |
| WER | Word Error Rate |

CER measures character-level mistakes.  
WER measures word-level mistakes.

Lower values are better.

---

## Current Status

Implemented:

- Project structure
- CRNN model architecture
- Greedy CTC decoder
- OCR metrics
- Image preprocessing module
- Dataset loader
- Character vocabulary utilities
- Streamlit interface skeleton

In progress:

- IAM dataset preparation script
- Training loop
- Evaluation script
- Full OCR inference in the Streamlit app
- Sample predictions and result visualization

---

## Example Use Case

This project can be used as a foundation for:

- Digitizing handwritten notes
- Reading handwritten forms
- OCR systems for scanned documents
- Document automation workflows
- Educational OCR research projects

---

## Future Improvements

- Add beam search decoding
- Add data augmentation
- Add full-page text line segmentation
- Add experiment tracking
- Add model training results
- Add sample predictions
- Deploy the Streamlit demo online
- Compare CRNN with transformer-based OCR models

---

## Author

Developed by [Fidan6557](https://github.com/Fidan6557)

---

## License

This project is intended for educational and portfolio purposes.