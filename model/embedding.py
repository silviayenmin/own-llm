"""Token and Positional Embedding layers for Mini GPT.

These modules map discrete input token IDs and sequence positions to continuous representation vectors.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from model.config import ModelConfig


class GPTTokenEmbedding(nn.Module):
    """Custom Token Embedding layer mapping vocab indices to continuous vectors."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the Token Embedding layer.

        Args:
            config: ModelConfig containing vocab_size and n_embd parameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Token Embedding layer.

        Args:
            x: Integer tensor of shape (B, T) containing token IDs.

        Returns:
            Continuous representation tensor of shape (B, T, n_embd).
        """
        return cast(torch.Tensor, self.embedding(x))


class GPTPositionalEmbedding(nn.Module):
    """Custom Positional Embedding layer mapping sequence indices to learned continuous vectors."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the Positional Embedding layer.

        Args:
            config: ModelConfig containing block_size and n_embd parameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, x: torch.Tensor | int) -> torch.Tensor:
        """Forward pass of the Positional Embedding layer.

        Args:
            x: Either an input tensor of shape (B, T) to match position lengths,
               or an integer sequence length T.

        Returns:
            If input is a tensor, returns shape (1, T, n_embd) for batch broadcasting.
            If input is an int, returns shape (T, n_embd).
        """
        if isinstance(x, torch.Tensor):
            T = x.size(-1)
            # Create position index tensor on the same device as the input tensor
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            # Unsqueeze to output shape (1, T, n_embd) for easy broadcasting over batch dimension
            return cast(torch.Tensor, self.embedding(pos).unsqueeze(0))
        else:
            T = x
            if T <= 0:
                raise ValueError(f"Sequence length T must be positive, got: {T}")
            # Create position index tensor on default device
            pos = torch.arange(0, T, dtype=torch.long)
            return cast(torch.Tensor, self.embedding(pos))
