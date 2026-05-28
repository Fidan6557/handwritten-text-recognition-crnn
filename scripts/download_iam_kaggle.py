"""
Download the IAM Handwriting Dataset via the Kaggle CLI.

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Place kaggle.json in ~/.kaggle/kaggle.json
       (Get it from https://www.kaggle.com/settings -> API -> Create New Token)

Usage:
    python scripts/download_iam_kaggle.py
"""

import subprocess
import sys
from pathlib import Path


KAGGLE_DATASET = "nibinv23/iam-handwriting-word-database"
OUTPUT_DIR = Path("data/raw/iam")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IAM dataset from Kaggle: {KAGGLE_DATASET}")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(OUTPUT_DIR),
            "--unzip",
        ],
        check=False,
    )

    if result.returncode != 0:
        print("\nDownload failed. Make sure kaggle.json is set up correctly.")
        sys.exit(1)

    print(f"\nDownload complete. Files saved to: {OUTPUT_DIR}")
    print("Now run: make prepare-iam")


if __name__ == "__main__":
    main()
