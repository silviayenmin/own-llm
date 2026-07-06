"""Custom Multi-Head Self-Attention layer implemented from scratch using basic PyTorch tensor operations.

This module implements the causal self-attention mechanism with causal masking, multi-head splitting,
scaled dot-product calculation, and dropout regularization.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

from model.config import ModelConfig


class GPTSelfAttention(nn.Module):
    """Custom Causal Multi-Head Self-Attention layer implemented from scratch."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the causal self-attention layer.

        Args:
            config: ModelConfig containing hyperparameters.
        """
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"Embedding dimension n_embd ({config.n_embd}) must be divisible "
                f"by n_head ({config.n_head})."
            )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Key, query, value projections for all heads combined into one linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer to prevent attending to future tokens
        # Tril creates lower-triangular matrix, view reshapes to match batch and head dimensions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the causal self-attention layer.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Attention output tensor of shape (B, T, n_embd).
        """
        B, T, C = x.size()

        # Compute query, key, value projections
        qkv = self.c_attn(x)  # Shape: (B, T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, nh, T, hs)
        # hs: head_size = C // n_head
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # Calculate self-attention scores: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

        # Apply causal mask: fill upper triangle with -inf
        bias = cast(torch.Tensor, self.bias)
        att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))

        # Compute attention probabilities and apply dropout
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Compute output vectors: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # Concatenate heads back: transpose to (B, T, nh, hs) -> contiguous -> view (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and residual dropout
        out = self.resid_dropout(self.c_proj(y))

        return cast(torch.Tensor, out)
