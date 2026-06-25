"""1D CNN for multichannel physiological-window classification.

The input is a window of shape (channels, time) -- by default 4 channels
(EDA, TEMP, HR, ACC motion) over a 30 s window at 4 Hz, i.e. (4, 120). A small
stack of Conv1d blocks learns local temporal patterns, global average pooling
collapses the time axis so the model is robust to small length changes, and a
linear head produces the four class logits.

A 1D CNN (rather than an RNN/Transformer) is a deliberate fit for the data: the
discriminative cues here are local, translation-invariant shapes -- EDA rises,
the level of motion intensity, sustained vs. spiky accelerometer activity -- and
a compact CNN learns those from a few thousand windows without overfitting.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class StressCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 conv_channels: List[int] = (32, 64, 128),
                 kernel_size: int = 7, dropout: float = 0.3) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_channels
        for out in conv_channels:
            layers += [
                nn.Conv1d(prev, out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            prev = out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool over time
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev, prev),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(prev, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def build_model(cfg: dict, in_channels: int, num_classes: int) -> StressCNN:
    m = cfg["model"]
    return StressCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_channels=list(m["conv_channels"]),
        kernel_size=m["kernel_size"],
        dropout=m["dropout"],
    )


if __name__ == "__main__":
    # Quick shape sanity check.
    net = StressCNN(in_channels=4, num_classes=4)
    dummy = torch.randn(8, 4, 120)
    out = net(dummy)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"output shape: {tuple(out.shape)}  |  parameters: {n_params:,}")
