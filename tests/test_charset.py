import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.charset import build_charset, save_charset, load_charset, BLANK_TOKEN


def make_temp_csv(texts, tmp_path, name="data.csv"):
    path = tmp_path / name
    pd.DataFrame({"text": texts}).to_csv(path, index=False)
    return str(path)


def test_blank_token_is_index_zero(tmp_path):
    csv_path = make_temp_csv(["hello"], tmp_path)
    char_to_idx, _ = build_charset([csv_path])
    assert char_to_idx[BLANK_TOKEN] == 0


def test_all_chars_present(tmp_path):
    csv_path = make_temp_csv(["abc", "def"], tmp_path)
    char_to_idx, _ = build_charset([csv_path])
    for char in "abcdef":
        assert char in char_to_idx


def test_idx_to_char_inverse(tmp_path):
    csv_path = make_temp_csv(["hello world"], tmp_path)
    char_to_idx, idx_to_char = build_charset([csv_path])
    for char, idx in char_to_idx.items():
        assert idx_to_char[idx] == char


def test_save_and_load_charset(tmp_path):
    csv_path = make_temp_csv(["abcdef"], tmp_path)
    char_to_idx, _ = build_charset([csv_path])

    charset_path = tmp_path / "charset.json"
    save_charset(char_to_idx, charset_path)

    loaded_char_to_idx, loaded_idx_to_char = load_charset(charset_path)

    assert loaded_char_to_idx == char_to_idx


def test_multiple_csv_paths(tmp_path):
    csv1 = make_temp_csv(["abc"], tmp_path, "a.csv")
    csv2 = make_temp_csv(["xyz"], tmp_path, "b.csv")

    char_to_idx, _ = build_charset([csv1, csv2])

    for char in "abcxyz":
        assert char in char_to_idx


def test_charset_indices_unique(tmp_path):
    csv_path = make_temp_csv(["hello world"], tmp_path)
    char_to_idx, _ = build_charset([csv_path])

    indices = list(char_to_idx.values())
    assert len(indices) == len(set(indices))
