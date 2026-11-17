"""Perdida, ponderacion y prediccion para la tarea ordinal CORAL."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from src.config import TrainingConfig
from src.tasks.base import ClassificationTask, LossFunction


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convierte clases enteras a umbrales binarios acumulativos.

    Ejemplo:
    Si num_classes = 5 y la etiqueta es 2, el vector debe ser [1, 1, 0, 0].

    Formas:
    - labels: (batch_size,)
    - salida: (batch_size, num_classes - 1) float32
    """

    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels[:, None] > thresholds).float()


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    BCE con logits sobre los K-1 umbrales ordinales.

    Suma la perdida sobre los umbrales de cada muestra y promedia el batch.
    Si hay pesos, aplica a cada muestra el peso de su clase real.

    Formas:
    - logits: (batch_size, num_classes - 1)
    - labels: (batch_size,)
    - class_weights: (num_classes,) o None
    - salida: escalar
    """

    levels = labels_to_levels(labels, num_classes).to(dtype=logits.dtype)
    loss_per_threshold = F.binary_cross_entropy_with_logits(
        logits, levels, reduction="none"
    )
    loss_per_sample = loss_per_threshold.sum(dim=1)
    if class_weights is not None:
        loss_per_sample = loss_per_sample * class_weights[labels]
    return loss_per_sample.mean()


def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:
    """
    Pesos por numero efectivo de muestras (Cui et al., CVPR 2019):

        w_c = (1 - beta) / (1 - beta ** n_c)

    Normaliza los pesos a media 1. Para una clase ausente usa n_c = 1
    al calcular su peso, como en el Laboratorio 01.

    Formas:
    - labels: (N,)
    - salida: (num_classes,) float32
    """

    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


@torch.no_grad()
def logits_to_ordinal_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convierte logits CORAL en una clase entera.

    Cuenta las probabilidades acumuladas que superan el umbral indicado.

    Formas:
    - logits: (batch_size, K-1)
    - salida: (batch_size,) int64
    """

    probabilities = torch.sigmoid(logits)
    return (probabilities > threshold).sum(dim=1)


class CoralLoss:
    """Envoltorio invocable de coral_loss con K y pesos fijos por fold."""

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return coral_loss(logits, labels, self.num_classes, self.class_weights)


class CoralTask(ClassificationTask):
    """
    Clasificacion ordinal con K-1 umbrales acumulativos.

    Conecta el Trainer con la perdida, la prediccion y los pesos por fold.

    Formas:
        logits: (B, K-1)
        labels: (B,) int64 en 0 .. K-1
    """

    name = "coral"

    def build_criterion(
        self,
        y_train: np.ndarray,
        config: TrainingConfig,
        device: torch.device,
    ) -> LossFunction:
        class_weights = None
        if config.use_class_weights:
            class_weights = effective_number_weights(
                y_train, self.num_classes, beta=config.beta
            ).to(device)
        return CoralLoss(self.num_classes, class_weights)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits_to_ordinal_predictions(logits)
