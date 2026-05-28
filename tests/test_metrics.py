from src.utils.metrics import character_error_rate, word_error_rate


def test_cer_identical_strings():
    assert character_error_rate("hello", "hello") == 0.0


def test_cer_completely_different():
    assert character_error_rate("aaa", "bbb") == 1.0


def test_cer_empty_target_and_prediction():
    assert character_error_rate("", "") == 0.0


def test_cer_empty_target_nonempty_prediction():
    assert character_error_rate("abc", "") == 1.0


def test_cer_ignores_spaces():
    assert character_error_rate("a b", "ab") == 0.0


def test_wer_identical_strings():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_empty_target_and_prediction():
    assert word_error_rate("", "") == 0.0


def test_wer_one_word_wrong():
    result = word_error_rate("hello world", "hello earth")
    assert result == 1 / 2


def test_wer_completely_different():
    result = word_error_rate("foo bar", "aaa bbb")
    assert result == 1.0
