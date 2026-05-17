import torch

from src.inference.decoder import greedy_ctc_decode


def test_greedy_ctc_decode_removes_blanks_and_repeated_tokens():
    idx_to_char = {
        0: "<blank>",
        1: "a",
        2: "b",
    }

    predicted_indices = torch.tensor([
        [0],
        [1],
        [1],
        [0],
        [2],
        [2],
        [0],
    ])

    num_classes = 3
    time_steps, batch_size = predicted_indices.shape

    log_probs = torch.full(
        (time_steps, batch_size, num_classes),
        fill_value=-10.0,
    )

    for t in range(time_steps):
        log_probs[t, 0, predicted_indices[t, 0]] = 0.0

    decoded = greedy_ctc_decode(log_probs, idx_to_char)

    assert decoded == ["ab"]