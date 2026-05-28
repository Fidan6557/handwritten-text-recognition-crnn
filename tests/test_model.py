import torch
import pytest

from src.models.crnn import CRNN


def test_output_shape():
    model = CRNN(num_classes=30, hidden_size=64, num_lstm_layers=1)
    model.eval()

    images = torch.randn(2, 1, 64, 512)

    with torch.no_grad():
        output = model(images)

    time_steps, batch_size, num_classes = output.shape
    assert batch_size == 2
    assert num_classes == 30
    assert time_steps > 0


def test_output_is_log_softmax():
    model = CRNN(num_classes=10, hidden_size=64, num_lstm_layers=1)
    model.eval()

    images = torch.randn(1, 1, 64, 512)

    with torch.no_grad():
        log_probs = model(images)

    probs_sum = log_probs.exp().sum(dim=2)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_different_batch_sizes():
    model = CRNN(num_classes=20, hidden_size=64, num_lstm_layers=1)
    model.eval()

    for batch_size in [1, 4, 8]:
        images = torch.randn(batch_size, 1, 64, 512)
        with torch.no_grad():
            output = model(images)
        assert output.shape[1] == batch_size
