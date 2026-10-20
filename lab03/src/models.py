"""
Neural network architectures for Lab 3:
1. Multi-Layer Perceptron (MLP)
2. Convolutional Neural Network (CNN)
3. Patch-Based Vision Transformer (AttentionModel)

Supports both:
- 'specified': The exact architecture dimensions stated in the manual bullet points.
- 'calibrated': Minimally adjusted architectures targeting 100,000–150,000 parameters.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Fashion-MNIST classification.
    
    Specified architecture:
        Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10
        Total parameters: 235,146
    
    Calibrated architecture (~100k-150k target):
        Flatten 784 -> Linear 144 -> ReLU -> Linear 72 -> ReLU -> Linear 10
        Total parameters: 124,210
    """
    def __init__(self, hidden_dim1: int = 256, hidden_dim2: int = 128, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """
    Convolutional Neural Network for Fashion-MNIST classification.
    
    Specified architecture:
        Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool2d(2) ->
        Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear(32*7*7, 10)
        Total parameters: 20,490
    
    Calibrated architecture (~100k-150k target):
        Conv2d(1, 64, 3, padding=1) -> ReLU -> MaxPool2d(2) ->
        Conv2d(64, 128, 3, padding=1) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear(128*7*7, 10)
        Total parameters: 137,226
    """
    def __init__(self, c1: int = 16, c2: int = 32, num_classes: int = 10):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c2 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, 28, 28)
        x = self.pool1(self.relu1(self.conv1(x)))  # -> (B, c1, 14, 14)
        x = self.pool2(self.relu2(self.conv2(x)))  # -> (B, c2, 7, 7)
        x = self.flatten(x)                        # -> (B, c2 * 49)
        x = self.fc(x)                             # -> (B, 10)
        return x


class AttentionModel(nn.Module):
    """
    Patch-based Vision Transformer for Fashion-MNIST classification.
    
    Specified architecture:
        - Image split into sixteen 7x7 patches (grid 4x4)
        - Patch linear projection to d_model=64
        - Learnable 1D position embeddings (16, 64)
        - 2 Transformer encoder layers (d_model=64, nhead=4, dim_feedforward=128)
        - Mean pooling across the 16 patch tokens
        - Linear head: 64 -> 10
        Total parameters: 71,818 (with pos_embed) / 70,794 (without pos_embed)
    
    Calibrated architecture (~100k-150k target):
        - Same patch projection and dimensions, with num_layers=4
        Total parameters: 138,762 (with pos_embed)
    """
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        patch_size: int = 7,
        num_classes: int = 10,
        use_pos_embed: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.use_pos_embed = use_pos_embed
        
        # 28x28 split into 16 patches of 7x7:
        # Conv2d(1, d_model, kernel_size=7, stride=7) executes the patch extraction
        # and linear projection in one equivalent, standard operation.
        self.patch_proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (28 // patch_size) * (28 // patch_size)  # 4 * 4 = 16
        self.num_patches = num_patches
        
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="relu",
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        # Patch projection: (B, d_model, 4, 4)
        x = self.patch_proj(x)
        # Flatten spatial dims to sequence: (B, d_model, 16) -> (B, 16, d_model)
        x = x.flatten(2).transpose(1, 2)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        x = self.transformer(x)    # (B, 16, d_model)
        x = x.mean(dim=1)          # Mean pool over sequence -> (B, d_model)
        x = self.head(x)           # (B, 10)
        return x


def get_model(name: str, mode: str = "specified") -> nn.Module:
    """
    Factory function returning model instances.

    Args:
        name: One of ['mlp', 'cnn', 'attention'].
        mode: 'specified' (exact lab manual specification) or
              'calibrated' (adjusted to ~100k-150k parameters).
    """
    name = name.lower()
    if mode == "specified":
        if name == "mlp":
            return MLP(hidden_dim1=256, hidden_dim2=128)
        elif name == "cnn":
            return CNN(c1=16, c2=32)
        elif name == "attention":
            return AttentionModel(d_model=64, nhead=4, dim_feedforward=128, num_layers=2)
        else:
            raise ValueError(f"Unknown model name: {name}")
    elif mode == "calibrated":
        if name == "mlp":
            return MLP(hidden_dim1=144, hidden_dim2=72)
        elif name == "cnn":
            return CNN(c1=64, c2=128)
        elif name == "attention":
            return AttentionModel(d_model=64, nhead=4, dim_feedforward=128, num_layers=4)
        else:
            raise ValueError(f"Unknown model name: {name}")
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'specified' or 'calibrated'.")


def get_all_models(mode: str = "specified") -> Dict[str, nn.Module]:
    """Returns a dictionary containing all three models for the given mode."""
    return {
        "MLP": get_model("mlp", mode=mode),
        "CNN": get_model("cnn", mode=mode),
        "Attention": get_model("attention", mode=mode),
    }
