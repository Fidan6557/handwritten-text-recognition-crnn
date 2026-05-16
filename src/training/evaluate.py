from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_iam_lines_file(lines_txt_path: str, images_dir: str) -> pd.DataFrame:
    """
    Parse IAM lines.txt annotation file.

    IAM lines.txt format contains metadata and the transcription text.
    Lines starting with # are comments and are ignored.

    Expected output:
    image_path,text
    """

    lines_txt_path = Path(lines_txt_path)
    images_dir = Path(images_dir)

    records = []

    with open(lines_txt_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(" ")

            image_id = parts[0]
            status = parts[1]

            if status != "ok":
                continue

            text = " ".join(parts[8:])
            text = text.replace("|", " ")

            folder_1 = image_id.split("-")[0]
            folder_2 = "-".join(image_id.split("-")[:2])

            image_path = images_dir / folder_1 / folder_2 / f"{image_id}.png"

            if image_path.exists():
                records.append(
                    {
                        "image_path": str(image_path),
                        "text": text,
                    }
                )

    return pd.DataFrame(records)


def create_train_val_test_split(
    dataframe: pd.DataFrame,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    Split dataset into train, validation and test CSV files.
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        dataframe,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
    )

    val_size_adjusted = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")


def main():
    lines_txt_path = "data/raw/iam/lines.txt"
    images_dir = "data/raw/iam/lines"
    output_dir = "data/processed"

    dataframe = parse_iam_lines_file(
        lines_txt_path=lines_txt_path,
        images_dir=images_dir,
    )

    if dataframe.empty:
        raise RuntimeError(
            "No valid samples found. Please check IAM dataset paths."
        )

    create_train_val_test_split(
        dataframe=dataframe,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()