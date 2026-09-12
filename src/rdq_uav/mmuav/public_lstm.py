"""Public MMUAV 9D ordinary-LSTM baseline.

The architecture and augmentation choices mirror the public training code at
dtc111111/Multi-Modal-UAV commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7.
This is not the paper's unavailable 7D Attention-LSTM implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class PublicLSTMClassifier(nn.Module):
    """LSTM(9,64,1) -> final time step -> Linear(64,2)."""

    def __init__(
        self, input_size: int = 9, hidden_size: int = 64,
        num_layers: int = 1, num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size,
            device=x.device, dtype=x.dtype,
        )
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def public_augment_batch(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Apply the public original/reverse/random-zero 0.5/0.25/0.25 policy.

    A clone is returned so diagnostic/evaluation tensors cannot be mutated.
    """
    result = x.clone()
    for index in range(result.shape[0]):
        choice = rng.choice(3, p=(0.50, 0.25, 0.25))
        if choice == 1:
            result[index] = torch.flip(result[index], dims=(0,))
        elif choice == 2:
            replace_count = int(rng.integers(1, 4))
            indices = rng.choice(result.shape[1], replace_count, replace=False)
            result[index, torch.as_tensor(indices, dtype=torch.long)] = 0
    return result


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    positive_predictions: int
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> BinaryMetrics:
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError(f"Invalid target/prediction shapes: {target.shape}, {prediction.shape}")
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    tp = int(np.sum((target == 1) & (prediction == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        accuracy=(tp + tn) / target.size,
        precision=precision,
        recall=recall,
        f1=f1,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        true_positive=tp,
        positive_predictions=int(np.sum(prediction == 1)),
        samples=int(target.size),
    )
