import argparse
from pathlib import Path

import torch

from src.data.segmentation import segment_text_lines, save_segmented_lines
from src.inference.predict import predict_text


def predict_page_text(
    image,
    model_path="outputs/checkpoints/best_crnn.pth",
    charset_path="outputs/charset.json",
    device=None,
):
    """
    Predict text from a full handwritten page image.

    Steps:
    1. Segment page into line images
    2. Predict text for each line
    3. Join line predictions into final text
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(model_path)
    charset_path = Path(charset_path)

    if not model_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Train the CRNN model first."
        )

    if not charset_path.exists():
        raise FileNotFoundError(
            "Charset file not found. Train the CRNN model first."
        )

    line_images = segment_text_lines(image)

    if not line_images:
        return ""

    predictions = []

    for line_image in line_images:
        prediction = predict_text(
            image=line_image,
            model_path=model_path,
            charset_path=charset_path,
            device=device,
        )

        predictions.append(prediction)

    return "\n".join(predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Predict handwritten text from a full page image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to handwritten page image.",
    )

    parser.add_argument(
        "--model",
        default="outputs/checkpoints/best_crnn.pth",
        help="Path to trained CRNN checkpoint.",
    )

    parser.add_argument(
        "--charset",
        default="outputs/charset.json",
        help="Path to charset JSON file.",
    )

    parser.add_argument(
        "--save-lines",
        action="store_true",
        help="Save segmented line crops for debugging.",
    )

    args = parser.parse_args()

    if args.save_lines:
        saved_paths = save_segmented_lines(
            image=args.image,
            output_dir="outputs/segmented_lines",
        )

        print("Saved segmented lines:")
        for path in saved_paths:
            print(path)

    predicted_text = predict_page_text(
        image=args.image,
        model_path=args.model,
        charset_path=args.charset,
    )

    print("Predicted Page Text:")
    print(predicted_text)


if __name__ == "__main__":
    main()