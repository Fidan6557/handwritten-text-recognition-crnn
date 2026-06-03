import pandas as pd

from src.utils.charset import BLANK_TOKEN, build_charset, load_charset, save_charset


def _make_csv(tmp_path, name, texts):
    path = tmp_path / name
    pd.DataFrame({"text": texts}).to_csv(path, index=False)
    return str(path)


def test_blank_token_at_index_zero(tmp_path):
    csv = _make_csv(tmp_path, "data.csv", ["hello", "world"])
    char_to_idx, idx_to_char = build_charset([csv])
    assert char_to_idx[BLANK_TOKEN] == 0
    assert idx_to_char[0] == BLANK_TOKEN


def test_all_characters_present(tmp_path):
    csv = _make_csv(tmp_path, "data.csv", ["abc"])
    char_to_idx, _ = build_charset([csv])
    for ch in "abc":
        assert ch in char_to_idx


def test_indices_are_unique(tmp_path):
    csv = _make_csv(tmp_path, "data.csv", ["abcdef"])
    char_to_idx, _ = build_charset([csv])
    indices = list(char_to_idx.values())
    assert len(indices) == len(set(indices))


def test_merge_multiple_csv_files(tmp_path):
    csv1 = _make_csv(tmp_path, "a.csv", ["abc"])
    csv2 = _make_csv(tmp_path, "b.csv", ["xyz"])
    char_to_idx, _ = build_charset([csv1, csv2])
    for ch in "abcxyz":
        assert ch in char_to_idx


def test_save_and_load_roundtrip(tmp_path):
    csv = _make_csv(tmp_path, "data.csv", ["hello world"])
    char_to_idx, _ = build_charset([csv])

    charset_path = tmp_path / "charset.json"
    save_charset(char_to_idx, charset_path)

    loaded_char_to_idx, loaded_idx_to_char = load_charset(charset_path)

    assert set(loaded_char_to_idx.keys()) == set(char_to_idx.keys())
    assert loaded_char_to_idx[BLANK_TOKEN] == 0
