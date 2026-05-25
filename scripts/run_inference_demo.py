"""
Quick inference demo on sample images.

Usage:
    python scripts/run_inference_demo.py --image sample_images/page_sample.jpg

Requires a trained model at outputs/checkpoints/best_crnn.pth
"""

import argparse
from pathlib import Path

from src.inference.predict_page import predict_page_text


def main():
    parser = argparse.ArgumentParser(description="Run OCR inference demo.")

    parser.add_argument(
        "--image",
        required=True,
        help="Path to handwritten image (line or full page).",
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

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"Running OCR on: {image_path}")

    predicted_text = predict_page_text(
        image=str(image_path),
        model_path=args.model,
        charset_path=args.charset,
    )

    print("\n--- Predicted Text ---")
    print(predicted_text)
    print("----------------------")


if __name__ == "__main__":
    main()
