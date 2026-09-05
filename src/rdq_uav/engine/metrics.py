from __future__ import annotations

from typing import Any

import torch


class ClassificationMetrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = logits.argmax(dim=1).detach().cpu().to(torch.int64)
        targets = targets.detach().cpu().to(torch.int64)
        self.update_predictions(predictions, targets)

    @torch.no_grad()
    def update_predictions(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = predictions.detach().cpu().to(torch.int64)
        targets = targets.detach().cpu().to(torch.int64)
        valid = (targets >= 0) & (targets < self.num_classes)
        indices = targets[valid] * self.num_classes + predictions[valid]
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion.to(torch.float64)
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
        recall = torch.where(support > 0, true_positive / support, 0.0)
        denominator = precision + recall
        f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
        total = matrix.sum()
        accuracy = true_positive.sum() / total if total > 0 else torch.tensor(0.0)
        valid_classes = support > 0
        return {
            "accuracy": float(accuracy),
            "macro_precision": float(precision[valid_classes].mean()) if valid_classes.any() else 0.0,
            "macro_recall": float(recall[valid_classes].mean()) if valid_classes.any() else 0.0,
            "macro_f1": float(f1[valid_classes].mean()) if valid_classes.any() else 0.0,
            "balanced_accuracy": float(recall[valid_classes].mean()) if valid_classes.any() else 0.0,
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "support": support.to(torch.int64).tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }
