"""
models/cnn_baseline.py

Standard CNN classifier used as the comparison baseline.
Architecture follows the paper: Conv2D(32, 3x3) -> ReLU -> MaxPool, repeated,
then Flatten -> Dense for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class CNNClassifier(nn.Module):
    """
    Baseline CNN for binary real/fake image classification.

    Paper specifies: 32 filters of 3x3, ReLU, MaxPool repeated pattern,
    ending with flatten + dense + binary output.
    """

    def __init__(self, in_channels=3, img_size=config.IMG_SIZE,
                 dropout=config.DROPOUT):
        super().__init__()

        # feature extraction
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # figure out flattened dim dynamically — avoids hardcoding
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            flat_dim = self.features(dummy).flatten(1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),    # binary output; use BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)    # [B]


def build_cnn(device=None, dropout=config.DROPOUT):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CNNClassifier(dropout=dropout).to(device)
