"""Controlled 9D Attention-LSTM reconstruction for the MMUAV experiment."""
from __future__ import annotations

import torch
from torch import nn


class AttentionLSTMClassifier(nn.Module):
    """LSTM(9,64,1) with scalar attention over all 20 hidden states.

    This M1 model changes only hidden-state aggregation relative to the public
    M0 classifier. The feature input, recurrent dimensions, and Linear(64,2)
    classification head are unchanged.
    """

    def __init__(
        self, input_size: int = 9, hidden_size: int = 64,
        num_layers: int = 1, num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Declaration order deliberately keeps M0 LSTM and classifier initial
        # parameters identical under the same seed.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size,
            device=x.device, dtype=x.dtype,
        )
        c0 = torch.zeros_like(h0)
        hidden_states, _ = self.lstm(x, (h0, c0))
        scores = self.attention(hidden_states)
        attention_weights = torch.softmax(scores, dim=1).squeeze(-1)
        context = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return self.fc(context), attention_weights
