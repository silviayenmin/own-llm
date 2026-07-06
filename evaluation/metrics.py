"""Model evaluation metrics for ownllm.

Provides computation functions for validation loss, perplexity, and token-level prediction accuracy.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_perplexity(loss: float) -> float:
    """Calculate the perplexity of the model given the cross-entropy loss.

    Perplexity is computed as e^(loss). A safety clipping is applied to prevent
    overflow errors.

    Args:
        loss: Scalar cross-entropy loss value.

    Returns:
        Perplexity value.
    """
    try:
        # Clip loss to avoid mathematical overflow when calculating exponent
        return math.exp(min(loss, 50.0))
    except OverflowError:
        return float("inf")


def calculate_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
) -> float:
    """Calculate the top-1 next-token prediction accuracy.

    Args:
        logits: Output logits of shape (B, T, vocab_size) or (B * T, vocab_size).
        targets: Target token IDs of shape (B, T) or (B * T,).
        ignore_index: Target token value to mask out/ignore in accuracy computation.

    Returns:
        Top-1 token prediction accuracy ratio between 0.0 and 1.0.
    """
    preds = torch.argmax(logits, dim=-1)

    # Flatten predictions and targets to calculate matches cleanly
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # Mask to ignore special padding or label tokens
    mask = targets_flat != ignore_index
    correct = (preds_flat == targets_flat) & mask

    total = mask.sum().item()
    if total == 0:
        return 0.0

    return correct.sum().item() / total


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation evaluation over a dataset split.

    Args:
        model: The GPT model to evaluate.
        dataloader: DataLoader containing evaluation batches.
        device: Device backend to place data on.
        max_batches: Optional limit on the number of batches to process.

    Returns:
        A dictionary containing average 'loss', 'perplexity', and 'accuracy'.
    """
    was_training = model.training
    model.eval()

    device_obj = torch.device(device)
    total_loss = 0.0
    total_accuracy = 0.0
    count = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            x, y = x.to(device_obj), y.to(device_obj)
            logits, loss = model(x, y)
            loss = cast(torch.Tensor, loss)

            total_loss += loss.item()
            acc = calculate_accuracy(logits, y)
            total_accuracy += acc
            count += 1

    # Restore original training state
    if was_training:
        model.train()

    if count == 0:
        return {"loss": 0.0, "perplexity": 1.0, "accuracy": 0.0}

    avg_loss = total_loss / count
    avg_accuracy = total_accuracy / count
    perplexity = calculate_perplexity(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": avg_accuracy,
    }
