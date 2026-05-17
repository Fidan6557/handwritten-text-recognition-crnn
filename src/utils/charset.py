import json
from pathlib import Path

import pandas as pd


BLANK_TOKEN = "<blank>"


def build_charset(csv_paths):
    """
    Build character vocabulary from one or more CSV files.

    CSV files must contain a 'text' column.
    """

    characters = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        for text in df["text"].astype(str):
            for char in text:
                characters.add(char)

    characters = sorted(characters)

    char_to_idx = {BLANK_TOKEN: 0}
    idx_to_char = {0: BLANK_TOKEN}

    for idx, char in enumerate(characters, start=1):
        char_to_idx[char] = idx
        idx_to_char[idx] = char

    return char_to_idx, idx_to_char


def save_charset(char_to_idx, path):
    """
    Save character vocabulary to JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(char_to_idx, file, ensure_ascii=False, indent=4)


def load_charset(path):
    """
    Load character vocabulary from JSON.
    """

    with open(path, "r", encoding="utf-8") as file:
        char_to_idx = json.load(file)

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return char_to_idx, idx_to_char