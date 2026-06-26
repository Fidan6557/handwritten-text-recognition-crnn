import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.crnn import CRNN
from src.inference.decoder import greedy_ctc_decode
from src.utils.charset import load_charset


DEFAULT_MODEL_PATH = "outputs/checkpoints/best_crnn.pth"
DEFAULT_CHARSET_PATH = "outputs/charset.json"


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


def parse_args():
    parser = argparse.ArgumentParser(description="Predict text from a single handwritten line image.")
    parser.add_argument("--image", required=True, help="Path to a handwritten line image.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained CRNN checkpoint.",
    )
    parser.add_argument(
        "--charset",
        default=DEFAULT_CHARSET_PATH,
        help="Path to charset JSON file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use. Defaults to cuda when available, otherwise cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    charset_path = Path(args.charset)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if not charset_path.exists():
        raise FileNotFoundError(f"Charset file not found: {charset_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prediction = predict_text(
        image=image_path,
        model_path=model_path,
        charset_path=charset_path,
        device=device,
    )

    print(prediction)


if __name__ == "__main__":
    main()
