"""Custom Layer Normalization implemented from scratch using fundamental PyTorch tensor operations.

This implementation allows disabling the learnable bias parameter to match modern optimizations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GPTLayerNorm(nn.Module):
    """Custom Layer Normalization implemented from scratch using basic PyTorch operations."""

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        """Initialize the Layer Normalization layer.

        Args:
            ndim: Feature dimensionality (e.g. n_embd).
            bias: Whether to include a learnable bias parameter.
            eps: Small constant for numerical stability to prevent division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Layer Normalization layer.

        Args:
            x: Input tensor of shape (..., ndim).

        Returns:
            Normalized tensor of shape (..., ndim).
        """
        # Compute mean along the last feature dimension
        mean = x.mean(-1, keepdim=True)
        # Compute population variance along the last feature dimension (unbiased=False)
        var = x.var(-1, keepdim=True, unbiased=False)

        # Normalize inputs
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift using weights and optional bias
        out = self.weight * x_norm
        if self.bias is not None:
            out = out + self.bias

        return out
