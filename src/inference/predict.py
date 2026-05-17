from pathlib import Path

import cv2
import torch
import numpy as np

from src.models.crnn import CRNN
from src.inference.decoder import greedy_ctc_decode
from src.utils.charset import load_charset


def preprocess_image_for_prediction(image, image_width=512, image_height=64):
    """
    Preprocess uploaded image for OCR prediction.
    """

    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image}")
    else:
        image = np.array(image)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (image_width, image_height))
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image

    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor


def load_ocr_model(model_path, charset_path, device="cpu"):
    """
    Load trained OCR model and character vocabulary.
    """

    char_to_idx, idx_to_char = load_charset(charset_path)

    model = CRNN(num_classes=len(char_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, idx_to_char


def predict_text(image, model_path, charset_path, device="cpu"):
    """
    Predict text from a handwritten text image.
    """

    model, idx_to_char = load_ocr_model(model_path, charset_path, device)

    image_tensor = preprocess_image_for_prediction(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        log_probs = model(image_tensor)
        prediction = greedy_ctc_decode(log_probs.cpu(), idx_to_char)[0]

    return prediction