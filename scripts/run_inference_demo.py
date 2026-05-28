"""
Quick CLI inference demo for the trained CRNN OCR model.

Supports both single line images and full handwritten page images.

Usage:
    python scripts/run_inference_demo.py --image sample_images/page_sample.jpg
    python scripts/run_inference_demo.py --image path/to/line.png --mode line
"""

import argparse
from pathlib import Path

import torch


DEFAULT_MODEL = "outputs/checkpoints/best_crnn.pth"
DEFAULT_CHARSET = "outputs/charset.json"


def main():
    parser = argparse.ArgumentParser(description="CRNN OCR inference demo.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--mode",
        choices=["line", "page"],
        default="page",
        help="OCR mode: 'line' for a single text line, 'page' for a full page (default: page).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to trained CRNN checkpoint.")
    parser.add_argument("--charset", default=DEFAULT_CHARSET, help="Path to charset JSON file.")
    args = parser.parse_args()

    model_path = Path(args.model)
    charset_path = Path(args.charset)

    if not model_path.exists():
        print(f"Model checkpoint not found: {model_path}")
        print("Train the model first with: make train")
        return

    if not charset_path.exists():
        print(f"Charset file not found: {charset_path}")
        print("Train the model first with: make train")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "line":
        from src.inference.predict import predict_text

        text = predict_text(
            image=args.image,
            model_path=args.model,
            charset_path=args.charset,
            device=device,
        )
        print(f"Predicted text: {text}")

    else:
        from src.inference.predict_page import predict_page_text

        text = predict_page_text(
            image=args.image,
            model_path=args.model,
            charset_path=args.charset,
            device=device,
        )
        print("Predicted page text:")
        print(text)


if __name__ == "__main__":
    main()
