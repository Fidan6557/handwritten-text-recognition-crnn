"""
Download IAM Handwriting Dataset from Kaggle.

Usage:
    python scripts/download_iam_kaggle.py

Requirements:
    - kaggle package installed
    - ~/.kaggle/kaggle.json configured with your API key

IAM dataset on Kaggle:
    https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
"""

import os
import subprocess
from pathlib import Path


def download_iam_from_kaggle(output_dir: str = "data/raw/iam"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading IAM Handwriting Dataset from Kaggle...")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "nibinv23/iam-handwriting-word-database",
            "--unzip",
            "-p",
            str(output_dir),
        ],
        check=True,
    )

    print(f"Dataset downloaded to: {output_dir}")


if __name__ == "__main__":
    download_iam_from_kaggle()
