from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def detect_columns(dataframe: pd.DataFrame):
    """
    Detect image and text columns from Hugging Face IAM parquet files.
    """

    possible_image_columns = ["image", "img", "line_image"]
    possible_text_columns = ["text", "transcription", "label", "sentence"]

    image_column = None
    text_column = None

    for column in possible_image_columns:
        if column in dataframe.columns:
            image_column = column
            break

    for column in possible_text_columns:
        if column in dataframe.columns:
            text_column = column
            break

    if image_column is None:
        raise ValueError(
            f"Could not detect image column. Available columns: {list(dataframe.columns)}"
        )

    if text_column is None:
        raise ValueError(
            f"Could not detect text column. Available columns: {list(dataframe.columns)}"
        )

    return image_column, text_column


def save_image_from_parquet_cell(image_data, output_path: Path):
    """
    Save image stored inside a parquet cell.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_data, dict):
        if image_data.get("bytes") is not None:
            image = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
            image.save(output_path)
            return

        if image_data.get("path") is not None:
            image = Image.open(image_data["path"]).convert("RGB")
            image.save(output_path)
            return

    if isinstance(image_data, Image.Image):
        image_data.convert("RGB").save(output_path)
        return

    raise TypeError(f"Unsupported image data type: {type(image_data)}")


def convert_split(parquet_path: Path, split_name: str, output_images_dir: Path):
    """
    Convert one parquet split into image files and a CSV file.
    """

    dataframe = pd.read_parquet(parquet_path)

    print(f"\n{split_name.upper()} SPLIT")
    print(f"Columns: {list(dataframe.columns)}")
    print(f"Samples: {len(dataframe)}")

    image_column, text_column = detect_columns(dataframe)

    records = []

    split_images_dir = output_images_dir / split_name
    split_images_dir.mkdir(parents=True, exist_ok=True)

    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Converting {split_name}"):
        text = str(row[text_column]).strip()

        if not text:
            continue

        image_path = split_images_dir / f"{split_name}_{index:06d}.png"

        save_image_from_parquet_cell(
            image_data=row[image_column],
            output_path=image_path,
        )

        records.append(
            {
                "image_path": str(image_path),
                "text": text,
            }
        )

    return pd.DataFrame(records)


def main():
    raw_dir = Path("data/raw/hf_iam")
    processed_dir = Path("data/processed")
    output_images_dir = processed_dir / "images"

    processed_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": raw_dir / "train.parquet",
        "val": raw_dir / "validation.parquet",
        "test": raw_dir / "test.parquet",
    }

    for split_name, parquet_path in split_files.items():
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing file: {parquet_path}. "
                "Place train.parquet, validation.parquet and test.parquet in data/raw/hf_iam/."
            )

        split_dataframe = convert_split(
            parquet_path=parquet_path,
            split_name=split_name,
            output_images_dir=output_images_dir,
        )

        output_csv = processed_dir / f"{split_name}.csv"
        split_dataframe.to_csv(output_csv, index=False)

        print(f"Saved {output_csv} with {len(split_dataframe)} samples.")

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()