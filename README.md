# Handwritten Text Recognition with CRNN, BiLSTM and CTC Loss

This project implements an end-to-end handwritten text recognition system using a Convolutional Recurrent Neural Network architecture.

The model combines CNN-based visual feature extraction, BiLSTM-based sequence modeling, and CTC Loss for alignment-free handwritten text recognition.

## Project Objective

The goal of this project is to recognize handwritten text from image inputs and convert it into machine-readable text.

This project focuses on line-level handwritten text recognition using the IAM Handwriting Dataset.

## Key Features

- Handwritten text image preprocessing
- CNN feature extractor
- BiLSTM sequence model
- CTC Loss for sequence alignment
- Greedy CTC decoding
- Character Error Rate and Word Error Rate evaluation
- Streamlit demo application
- Clean and reproducible project structure

## Model Architecture

```text
Input Image
    ↓
CNN Feature Extractor
    ↓
Feature Sequence
    ↓
BiLSTM Layers
    ↓
Linear Classifier
    ↓
CTC Decoder
    ↓
Predicted Text