from src.utils.metrics import character_error_rate, word_error_rate


def test_cer_perfect_match():
    assert character_error_rate("hello", "hello") == 0.0


def test_cer_empty_target():
    assert character_error_rate("", "") == 0.0


def test_cer_partial_error():
    cer = character_error_rate("helo", "hello")
    assert 0.0 < cer <= 1.0


def test_cer_completely_wrong():
    cer = character_error_rate("xyz", "abc")
    assert cer > 0.0


def test_wer_perfect_match():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_empty_target():
    assert word_error_rate("", "") == 0.0


def test_wer_one_wrong_word():
    wer = word_error_rate("hello earth", "hello world")
    assert wer == 0.5


def test_wer_completely_wrong():
    wer = word_error_rate("foo bar", "hello world")
    assert wer == 1.0


def test_cer_ignores_spaces():
    cer = character_error_rate("helloworld", "hello world")
    assert cer == 0.0
