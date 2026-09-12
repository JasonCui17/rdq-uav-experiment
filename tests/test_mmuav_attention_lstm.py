from __future__ import annotations

import unittest

import torch

from rdq_uav.mmuav.attention_lstm import AttentionLSTMClassifier
from rdq_uav.mmuav.public_lstm import PublicLSTMClassifier


class TestMMUAVAttentionLSTM(unittest.TestCase):
    def test_forward_shapes_and_probability_simplex(self) -> None:
        model = AttentionLSTMClassifier()
        logits, weights = model(torch.randn(5, 20, 9))
        self.assertEqual(tuple(logits.shape), (5, 2))
        self.assertEqual(tuple(weights.shape), (5, 20))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-6))

    def test_only_aggregation_parameters_are_added(self) -> None:
        torch.manual_seed(42)
        public = PublicLSTMClassifier()
        torch.manual_seed(42)
        attention = AttentionLSTMClassifier()
        for name, value in public.state_dict().items():
            self.assertTrue(torch.equal(value, attention.state_dict()[name]), name)
        extra = set(attention.state_dict()) - set(public.state_dict())
        self.assertEqual(extra, {"attention.weight", "attention.bias"})

    def test_backward_is_finite(self) -> None:
        model = AttentionLSTMClassifier()
        logits, _ = model(torch.randn(4, 20, 9))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
