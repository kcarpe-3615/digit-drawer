import torch
from torch import nn


class DigitClassifier(nn.Module):
    """
    A convolutional neural network for 28x28 grayscale digit images.

    Expected input shape:
        (batch_size, 1, 28, 28)

    Output shape:
        (batch_size, 10)
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Input: (batch, 1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Output: (batch, 32, 14, 14)
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Output: (batch, 64, 7, 7)
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 128 feature maps, each 7x7 pixels.
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(p=0.35),

            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        logits = self.classifier(features)

        return logits