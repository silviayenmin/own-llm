"""Standard Feed Forward MLP network block for Mini GPT.

This module implements a two-layer MLP with a hidden dimension expansion of 4x,
GELU activation function, contraction projection, and dropout regularization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import ModelConfig


class GPTFeedForward(nn.Module):
    """Custom Feed Forward MLP network block implemented from scratch."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the feedforward network block.

        Args:
            config: ModelConfig containing hyperparameters.
        """
        super().__init__()
        # Expansion layer: n_embd -> 4 * n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Activation function (GELU is standard in GPT architecture)
        self.gelu = nn.GELU()
        # Contraction/projection layer: 4 * n_embd -> n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Regularization dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feedforward MLP network.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
