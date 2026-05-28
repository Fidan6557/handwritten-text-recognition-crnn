"""
Download the IAM Handwriting Dataset from Hugging Face Hub.

Usage:
    python scripts/download_hf_iam.py
"""

from pathlib import Path


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    output_dir = Path("data/raw/hf_iam")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading IAM dataset from Hugging Face Hub...")
    dataset = load_dataset("Teklia/IAM-line")

    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    for split_name, hf_split in split_map.items():
        if hf_split not in dataset:
            print(f"Split '{hf_split}' not found, skipping.")
            continue

        out_path = output_dir / f"{split_name}.parquet"
        dataset[hf_split].to_parquet(str(out_path))
        print(f"Saved {out_path} ({len(dataset[hf_split])} samples)")

    print("\nDownload complete. Now run: make prepare-hf-iam")


if __name__ == "__main__":
    main()
