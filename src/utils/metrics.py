import editdistance


def character_error_rate(prediction: str, target: str) -> float:
    prediction = prediction.replace(" ", "")
    target = target.replace(" ", "")

    if len(target) == 0:
        return 0.0 if len(prediction) == 0 else 1.0

    return editdistance.eval(prediction, target) / len(target)


def word_error_rate(prediction: str, target: str) -> float:
    prediction_words = prediction.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 0.0 if len(prediction_words) == 0 else 1.0

    return editdistance.eval(prediction_words, target_words) / len(target_words)