import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import calculate_accuracy, calculate_perplexity, evaluate_model


class MockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple mock dataset returning random integers for model testing."""

    def __init__(self, vocab_size: int, size: int = 10, seq_len: int = 4) -> None:
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x, y


class MockModel(nn.Module):
    """Simple mock model that simulates GPT outputs for testing evaluation loops."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # Unused parameter to keep gradients/optimizer check happy if needed
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        # Outputs mock logits of shape (B, T, vocab_size)
        logits = torch.randn(B, T, self.vocab_size)
        loss = torch.tensor(2.5) if targets is not None else None
        return logits, loss


def test_perplexity() -> None:
    """Verify that calculate_perplexity computes exponential values correctly."""
    assert calculate_perplexity(0.0) == 1.0
    assert pytest.approx(calculate_perplexity(1.0)) == math.exp(1.0)

    # Check safety clipping against overflow
    assert calculate_perplexity(100.0) == math.exp(50.0)


def test_accuracy() -> None:
    """Verify calculate_accuracy calculates correct prediction ratio."""
    B, T, vocab_size = 2, 4, 10

    # Target values
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, -1, 8]])

    # Make predictions (by constructing logits where target index has the highest logit)
    logits = torch.zeros(B, T, vocab_size)

    # Set correct logits
    logits[0, 0, 1] = 10.0  # correct 1
    logits[0, 1, 2] = 10.0  # correct 2
    logits[0, 2, 9] = 10.0  # INCORRECT (target is 3, prediction is 9)
    logits[0, 3, 4] = 10.0  # correct 4

    logits[1, 0, 5] = 10.0  # correct 5
    logits[1, 1, 6] = 10.0  # correct 6
    logits[1, 2, 9] = 10.0  # (target is -1, ignored)
    logits[1, 3, 8] = 10.0  # correct 8

    # We have 7 non-ignored tokens:
    # Row 0: 3 correct, 1 incorrect
    # Row 1: 3 correct (one is target=-1, ignored)
    # Total non-ignored: 7
    # Correct non-ignored: 6
    # Accuracy should be 6 / 7 = 0.85714...
    acc = calculate_accuracy(logits, targets, ignore_index=-1)
    assert pytest.approx(acc) == 6 / 7


def test_accuracy_all_ignored() -> None:
    """Verify calculate_accuracy handles cases where all elements are ignored."""
    logits = torch.randn(1, 4, 10)
    targets = torch.tensor([[-1, -1, -1, -1]])
    assert calculate_accuracy(logits, targets, ignore_index=-1) == 0.0


def test_evaluate_model() -> None:
    """Verify evaluate_model runs validation passes on DataLoader correctly."""
    vocab_size = 10
    model = MockModel(vocab_size)
    dataset = MockDataset(vocab_size, size=6, seq_len=4)
    dataloader = DataLoader(dataset, batch_size=2)

    results = evaluate_model(model, dataloader, device="cpu", max_batches=2)

    # Check key mappings in dictionary
    assert "loss" in results
    assert "perplexity" in results
    assert "accuracy" in results

    # Verify that mock loss (2.5) propagates to output metrics
    assert results["loss"] == 2.5
    assert results["perplexity"] == math.exp(2.5)
    assert 0.0 <= results["accuracy"] <= 1.0
