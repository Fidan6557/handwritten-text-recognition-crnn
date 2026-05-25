"""
Download IAM Handwriting Dataset from Hugging Face Hub.

Usage:
    python scripts/download_hf_iam.py

Requirements:
    - datasets package installed (pip install datasets)
    - Hugging Face account and huggingface-cli login (if dataset is gated)

The script downloads the IAM dataset splits as parquet files into data/raw/hf_iam/.
Then run: python -m src.data.prepare_hf_iam
"""

from pathlib import Path


def download_hf_iam(output_dir: str = "data/raw/hf_iam"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the datasets package: pip install datasets"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading IAM dataset from Hugging Face Hub...")

    dataset = load_dataset("Teklia/IAM-line")

    for split_name in ["train", "validation", "test"]:
        split_key = split_name if split_name != "validation" else "validation"
        output_path = output_dir / f"{split_name}.parquet"

        dataset[split_key].to_parquet(str(output_path))
        print(f"Saved {split_name} split to: {output_path}")

    print("\nDownload complete.")
    print("Next step: python -m src.data.prepare_hf_iam")


if __name__ == "__main__":
    download_hf_iam()
