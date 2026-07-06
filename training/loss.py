"""Loss function utilities for ownllm.

Provides helper functions for calculating cross entropy sequence loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def calculate_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
) -> torch.Tensor:
    """Calculate the cross entropy loss between logits and targets.

    Args:
        logits: Output logits of shape (B, T, vocab_size) or (B * T, vocab_size).
        targets: Target token IDs of shape (B, T) or (B * T,).
        ignore_index: Target token value to ignore in loss calculation.

    Returns:
        Scalar cross entropy loss tensor.
    """
    if len(logits.shape) == 3:
        logits = logits.view(-1, logits.size(-1))
    if len(targets.shape) == 2:
        targets = targets.view(-1)

    return F.cross_entropy(logits, targets, ignore_index=ignore_index)
