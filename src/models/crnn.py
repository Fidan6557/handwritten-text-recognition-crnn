import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for handwritten text recognition.

    Architecture:
    CNN feature extractor -> BiLSTM sequence model -> Linear classifier
    """

    def __init__(self, num_classes: int, hidden_size: int = 256, num_lstm_layers: int = 2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=512 * 4,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.3,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)

        batch_size, channels, height, width = features.size()

        features = features.permute(3, 0, 1, 2)
        features = features.contiguous().view(width, batch_size, channels * height)

        sequence_output, _ = self.rnn(features)
        logits = self.classifier(sequence_output)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs