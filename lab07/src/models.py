"""
Model architectures for Lab 07:
1. BaselineCNN: Oversized CNN (~1,000,000 parameters) serving as the baseline and teacher.
2. StudentCNN: Compact CNN (~100,000 parameters) serving as the distillation target.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Oversized Convolutional Neural Network for Fashion-MNIST classification.
    Engineered to possess approximately 1,000,000 trainable parameters (~1M).
    
    Architecture:
        Conv1(1, c1, 3x3) -> ReLU
        Conv2(c1, c2, 3x3) -> ReLU -> MaxPool(2x2) [14x14]
        Conv3(c2, c3, 3x3) -> ReLU
        Conv4(c3, c4, 3x3) -> ReLU -> MaxPool(2x2) [7x7] -> AdaptiveAvgPool(1x1)
        Flatten
        FC1(c4, fc_dim) -> ReLU
        FC2(fc_dim, 10)
        
    With default hyperparameters (c1=64, c2=128, c3=240, c4=288, fc_dim=128):
        - Conv1: 1 * 64 * 9 + 64 = 640
        - Conv2: 64 * 128 * 9 + 128 = 73,856
        - Conv3: 128 * 240 * 9 + 240 = 276,720
        - Conv4: 240 * 288 * 9 + 288 = 622,368
        - FC1: 288 * 128 + 128 = 36,992
        - FC2: 128 * 10 + 10 = 1,290
        - Total parameters: 1,011,866 (~1.01M, 101.1% of 1M target)
    """

    def __init__(
        self,
        in_channels: int = 1,
        c1: int = 64,
        c2: int = 128,
        c3: int = 240,
        c4: int = 288,
        fc_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.fc_dim = fc_dim
        self.num_classes = num_classes

        # Feature extractor - Block 1
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Feature extractor - Block 2
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(c4, fc_dim)
        self.relu_fc = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, 28, 28)
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.relu1(self.conv1(x))
        x = self.pool1(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.pool3(self.pool2(self.relu4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Returns total number of parameters.
        If trainable_only=True, counts only parameters where requires_grad=True.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_prunable_weights(self) -> List[Tuple[str, nn.Parameter]]:
        """
        Returns list of (name, parameter) tuples for eligible pruning weights.
        Specifically targeting weights of Conv2d and Linear layers (biases excluded).
        """
        prunables = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunables.append((f"{name}.weight", module.weight))
        return prunables


class StudentCNN(nn.Module):
    """
    Compact Convolutional Neural Network for Fashion-MNIST classification.
    Engineered to possess approximately 100,000 trainable parameters (~100k).
    
    Architecture:
        Conv1(1, c1, 3x3) -> ReLU -> MaxPool(2x2) [14x14]
        Conv2(c1, c2, 3x3) -> ReLU -> MaxPool(2x2) [7x7]
        Conv3(c2, c3, 3x3) -> ReLU -> AdaptiveAvgPool(1x1)
        Flatten
        FC1(c3, fc_dim) -> ReLU
        FC2(fc_dim, 10)
        
    With default hyperparameters (c1=32, c2=64, c3=128, fc_dim=64):
        - Conv1: 1 * 32 * 9 + 32 = 320
        - Conv2: 32 * 64 * 9 + 64 = 18,496
        - Conv3: 64 * 128 * 9 + 128 = 73,856
        - FC1: 128 * 64 + 64 = 8,256
        - FC2: 64 * 10 + 10 = 650
        - Total parameters: 101,578 (~101.5k, 101.5% of 100k target)
    """

    def __init__(
        self,
        in_channels: int = 1,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        fc_dim: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.fc_dim = fc_dim
        self.num_classes = num_classes

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=False)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(c3, fc_dim)
        self.relu_fc = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, 28, 28)
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Returns total number of parameters.
        If trainable_only=True, counts only parameters where requires_grad=True.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_baseline_model(config: Optional[Dict[str, Any]] = None) -> BaselineCNN:
    """Factory helper to build BaselineCNN from configuration dictionary."""
    cfg = config or {}
    model_cfg = cfg.get("baseline_model", {})
    return BaselineCNN(
        in_channels=model_cfg.get("in_channels", 1),
        c1=model_cfg.get("c1", 64),
        c2=model_cfg.get("c2", 128),
        c3=model_cfg.get("c3", 240),
        c4=model_cfg.get("c4", 288),
        fc_dim=model_cfg.get("fc_dim", 128),
        num_classes=model_cfg.get("num_classes", 10),
    )


def build_student_model(config: Optional[Dict[str, Any]] = None) -> StudentCNN:
    """Factory helper to build StudentCNN from configuration dictionary."""
    cfg = config or {}
    model_cfg = cfg.get("student_model", {})
    return StudentCNN(
        in_channels=model_cfg.get("in_channels", 1),
        c1=model_cfg.get("c1", 32),
        c2=model_cfg.get("c2", 64),
        c3=model_cfg.get("c3", 128),
        fc_dim=model_cfg.get("fc_dim", 64),
        num_classes=model_cfg.get("num_classes", 10),
    )
