from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rdq_uav.mmuav.public_lstm import (
    PublicLSTMClassifier,
    binary_metrics,
    public_augment_batch,
)
from tools.build_mmuav_cluster_dataset import extract_window
from tools.make_mmuav_sequence_splits import make_splits, validate_disjoint
from tools.train_mmuav_cluster_classifier import EarlyStoppingState


class TestMMUAVPublicClassifier(unittest.TestCase):
    def test_early_stopping_after_fifteen_non_improvements(self) -> None:
        state = EarlyStoppingState(patience=15)
        improved, stopped = state.update(1.0, 1)
        self.assertTrue(improved)
        self.assertFalse(stopped)
        for epoch in range(2, 16):
            _, stopped = state.update(1.0, epoch)
            self.assertFalse(stopped)
        _, stopped = state.update(1.0, 16)
        self.assertTrue(stopped)
        self.assertEqual(state.counter, 15)
        self.assertEqual(state.best_epoch, 1)

    def test_early_stopping_improvement_resets_patience_and_best_epoch(self) -> None:
        state = EarlyStoppingState(patience=3)
        state.update(1.0, 1)
        state.update(1.1, 2)
        state.update(1.2, 3)
        improved, stopped = state.update(0.9, 4)
        self.assertTrue(improved)
        self.assertFalse(stopped)
        self.assertEqual(state.counter, 0)
        self.assertEqual(state.best_epoch, 4)
        self.assertEqual(state.best_loss, 0.9)
        for epoch in (5, 6):
            _, stopped = state.update(0.95, epoch)
            self.assertFalse(stopped)
        _, stopped = state.update(0.95, 7)
        self.assertTrue(stopped)
        self.assertEqual(state.best_epoch, 4)

    def test_early_stopping_min_delta(self) -> None:
        state = EarlyStoppingState(patience=2, min_delta=0.1)
        state.update(1.0, 1)
        improved, _ = state.update(0.95, 2)
        self.assertFalse(improved)
        improved, _ = state.update(0.89, 3)
        self.assertTrue(improved)
        self.assertEqual(state.best_epoch, 3)

    def test_sequence_splits_are_disjoint_and_reproducible(self) -> None:
        sequences = [f"seq{index:04d}" for index in range(1, 103)]
        first = make_splits(sequences, seed=42)
        second = make_splits(sequences, seed=42)
        self.assertEqual(first, second)
        validate_disjoint(first)
        self.assertEqual(len(first["train_sub"]), 72)
        self.assertEqual(len(first["validation_sub"]), 15)
        self.assertEqual(len(first["heldout_test_sub"]), 15)

    def test_public_lstm_shape_and_metrics(self) -> None:
        model = PublicLSTMClassifier()
        logits = model(torch.zeros(4, 20, 9))
        self.assertEqual(tuple(logits.shape), (4, 2))
        metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertEqual(metrics.true_positive, 1)
        self.assertAlmostEqual(metrics.f1, 0.5)

    def test_public_augmentation_preserves_shape_and_input(self) -> None:
        value = torch.arange(4 * 20 * 9, dtype=torch.float32).reshape(4, 20, 9)
        original = value.clone()
        augmented = public_augment_batch(value, np.random.default_rng(42))
        self.assertEqual(tuple(augmented.shape), (4, 20, 9))
        self.assertTrue(torch.equal(value, original))

    def test_extract_window_produces_20x9_and_public_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for frame in range(20):
                # Ten compact target points plus ten compact background points.
                target = np.tile(np.array([[1.0, 2.0, 3.0]]), (10, 1))
                background = np.tile(np.array([[20.0, 20.0, 20.0]]), (10, 1))
                data = np.concatenate((target, background, np.zeros((5, 3))))
                path = root / f"{1000.0 + frame * 0.1:.6f}.npy"
                np.save(path, data)
                paths.append(path)
            features, labels, rows, stats = extract_window(
                paths,
                np.array([1000.0]),
                np.array([[1.0, 2.0, 3.0]]),
                "train_sub", "seq0001", 0,
            )
            self.assertEqual(features.shape, (2, 20, 9))
            self.assertEqual(labels.shape, (2,))
            self.assertEqual(int(labels.sum()), 1)
            self.assertEqual(stats["positive"], 1)
            self.assertEqual(len(json.loads(rows[0]["frame_timestamps"])), 20)


if __name__ == "__main__":
    unittest.main()
