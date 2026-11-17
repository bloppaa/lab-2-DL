"""Conv1D + MLP con cabeza ordinal CORAL."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.blocks import Conv1DBackbone


class CoralLayer(nn.Module):
    """
    Produce K-1 logits acumulativos a partir de un puntaje latente
    compartido y sesgos decrecientes.

    Formas esperadas:
    - x: (batch_size, input_size)
    - salida: (batch_size, num_classes - 1)
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORAL requiere al menos dos clases.")

        self.input_size = input_size
        self.num_classes = num_classes
        self.shared_weight = nn.Linear(input_size, 1, bias=False)
        self.bias_base = nn.Parameter(torch.zeros(1))
        self.bias_deltas = nn.Parameter(torch.zeros(num_classes - 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scores = self.shared_weight(inputs)
        gaps = F.softplus(self.bias_deltas)
        offsets = torch.cat((self.bias_base.new_zeros(1), gaps)).cumsum(dim=0)
        biases = self.bias_base - offsets
        return scores + biases


class Conv1DCoralNet(nn.Module):
    """
    Conv1D + MLP con cabeza CORAL.

    El tronco acepta entradas (B, L) o (B, 1, L).
    La cabeza devuelve K-1 logits por muestra para la perdida ordinal.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Conv1DBackbone(
            num_features=num_features,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.coral = CoralLayer(self.backbone.output_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return self.coral(features)
