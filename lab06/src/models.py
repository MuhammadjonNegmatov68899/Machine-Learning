"""
Model architecture for Lab 6:
Reuses the best-performing architecture from Lab 3 (Convolutional Neural Network - CNN).
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network for Fashion-MNIST classification.
    Identified as the 'Cost-Effectiveness Champion' in Lab 3.
    
    Architecture:
        Conv2d(in_channels, c1, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2) ->
        Conv2d(c1, c2, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear(c2 * 7 * 7, num_classes)
        
    Default parameter count (c1=16, c2=32, num_classes=10):
        - Conv1: 1 * 16 * 3 * 3 + 16 = 160
        - Conv2: 16 * 32 * 3 * 3 + 32 = 4,640
        - Linear: (32 * 7 * 7) * 10 + 10 = 15,690
        - Total parameters: 20,490
    """

    def __init__(
        self,
        in_channels: int = 1,
        c1: int = 16,
        c2: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.c1 = c1
        self.c2 = c2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c2 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, 28, 28)
        Returns:
            Class logits of shape (B, num_classes)
        """
        x = self.pool1(self.relu1(self.conv1(x)))  # -> (B, c1, 14, 14)
        x = self.pool2(self.relu2(self.conv2(x)))  # -> (B, c2, 7, 7)
        x = self.flatten(x)                        # -> (B, c2 * 49)
        x = self.fc(x)                             # -> (B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(config: Dict[str, Any]) -> CNN:
    """
    Factory function to initialize model from configuration dictionary.
    """
    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 1)
    c1 = model_cfg.get("c1", 16)
    c2 = model_cfg.get("c2", 32)
    num_classes = model_cfg.get("num_classes", 10)

    return CNN(
        in_channels=in_channels,
        c1=c1,
        c2=c2,
        num_classes=num_classes,
    )
