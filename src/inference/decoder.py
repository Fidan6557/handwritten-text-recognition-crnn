import torch


def greedy_ctc_decode(log_probs, idx_to_char, blank_idx=0):
    """
    Decode CTC model outputs using greedy decoding.

    Args:
        log_probs: Tensor with shape [T, B, C]
        idx_to_char: Dictionary mapping class indices to characters
        blank_idx: CTC blank token index

    Returns:
        List of decoded text predictions
    """

    predictions = torch.argmax(log_probs, dim=2)
    predictions = predictions.permute(1, 0)

    decoded_texts = []

    for sequence in predictions:
        previous_idx = None
        decoded_chars = []

        for idx in sequence.tolist():
            if idx != blank_idx and idx != previous_idx:
                decoded_chars.append(idx_to_char[idx])

            previous_idx = idx

        decoded_texts.append("".join(decoded_chars))

    return decoded_texts