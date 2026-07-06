"""Transformer block for Mini GPT.

This module implements a Pre-Layer Normalization Transformer block combining
causal self-attention, feedforward MLP, and residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import GPTSelfAttention
from model.config import ModelConfig
from model.feedforward import GPTFeedForward
from model.layernorm import GPTLayerNorm


class GPTTransformerBlock(nn.Module):
    """Custom Transformer Block layer implemented from scratch."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the Transformer block.

        Args:
            config: ModelConfig containing hyperparameters.
        """
        super().__init__()
        # Pre-LN 1: Layer normalization before self-attention
        self.ln_1 = GPTLayerNorm(config.n_embd, bias=config.bias)
        # Multi-head self-attention module
        self.attn = GPTSelfAttention(config)
        # Pre-LN 2: Layer normalization before MLP
        self.ln_2 = GPTLayerNorm(config.n_embd, bias=config.bias)
        # Feedforward MLP block
        self.mlp = GPTFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer block.

        Applies Pre-Layer Normalization and residual connections.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        # Self-attention branch with residual connection
        x = x + self.attn(self.ln_1(x))
        # Feedforward branch with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x
