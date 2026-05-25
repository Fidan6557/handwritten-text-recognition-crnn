import torch

from src.models.crnn import CRNN


def test_crnn_output_shape():
    model = CRNN(num_classes=40)
    model.eval()

    batch_size = 2
    images = torch.randn(batch_size, 1, 64, 512)

    with torch.no_grad():
        output = model(images)

    assert output.dim() == 3
    assert output.shape[1] == batch_size
    assert output.shape[2] == 40


def test_crnn_log_softmax_output():
    model = CRNN(num_classes=30)
    model.eval()

    images = torch.randn(1, 1, 64, 512)

    with torch.no_grad():
        output = model(images)

    # log_softmax output must be <= 0
    assert (output <= 0).all()


def test_crnn_different_num_classes():
    for num_classes in [10, 50, 100]:
        model = CRNN(num_classes=num_classes)
        model.eval()

        images = torch.randn(1, 1, 64, 512)

        with torch.no_grad():
            output = model(images)

        assert output.shape[2] == num_classes


def test_crnn_custom_hidden_size():
    model = CRNN(num_classes=40, hidden_size=128, num_lstm_layers=1)
    model.eval()

    images = torch.randn(1, 1, 64, 512)

    with torch.no_grad():
        output = model(images)

    assert output.dim() == 3


def test_crnn_batch_consistency():
    model = CRNN(num_classes=40)
    model.eval()

    images = torch.randn(4, 1, 64, 512)

    with torch.no_grad():
        output = model(images)

    assert output.shape[1] == 4
