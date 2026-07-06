from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from model.config import GPTConfig
from tokenizer.tokenizer import CharTokenizer
from training.dataset import GPTDataset, preprocess_data


def test_preprocess_and_dataset(tmp_path: Path) -> None:
    """Verify preprocessing correctly serializes data and GPTDataset loads sequences properly."""
    # Setup temporary file paths
    raw_file = tmp_path / "raw.txt"
    processed_dir = tmp_path / "processed"

    # Write a simple text corpus
    text = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(text)

    # Initialize a simple config
    config = GPTConfig()
    config.data.raw_data_path = str(raw_file)
    config.data.processed_dir = str(processed_dir)
    config.data.train_split = 0.8  # 80% train, 20% val

    # Setup tokenizer (CharTokenizer for testing)
    chars = sorted(set(text))
    tokenizer = CharTokenizer(chars=chars)

    # Run preprocessing
    preprocess_data(config, tokenizer)

    # Verify output files exist
    train_bin_path = processed_dir / "train.bin"
    val_bin_path = processed_dir / "val.bin"
    assert train_bin_path.exists()
    assert val_bin_path.exists()

    # Calculate expected token lengths
    total_tokens = len(text)
    expected_train_len = int(total_tokens * config.data.train_split)
    expected_val_len = total_tokens - expected_train_len

    # Check file sizes (2 bytes per uint16 token)
    assert train_bin_path.stat().st_size == expected_train_len * 2
    assert val_bin_path.stat().st_size == expected_val_len * 2

    # Instantiate GPTDataset
    block_size = 4
    dataset = GPTDataset(data_path=train_bin_path, block_size=block_size)

    # Verify dataset length
    assert len(dataset) == (expected_train_len - 1) // block_size

    # Check __getitem__ values and types
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)

    # Verify shifted labels (y[t] == x[t + 1])
    # x should contain tokens 0, 1, 2, 3
    # y should contain tokens 1, 2, 3, 4
    for t in range(block_size - 1):
        assert y[t] == x[t + 1]


def test_dataset_invalid_block_size(tmp_path: Path) -> None:
    """Verify GPTDataset raises ValueError if dataset is too small for block_size."""
    bin_file = tmp_path / "tiny.bin"
    arr = np.array([1, 2, 3], dtype=np.uint16)
    arr.tofile(bin_file)

    # Block size of 3 requires at least 4 tokens (since y gets block_size + 1 elements)
    with pytest.raises(ValueError):
        GPTDataset(data_path=bin_file, block_size=3)


def test_dataset_dataloader_integration(tmp_path: Path) -> None:
    """Verify that wrapping GPTDataset in DataLoader returns batches of expected shape."""
    bin_file = tmp_path / "data.bin"
    # Create 20 tokens
    arr = np.array(list(range(20)), dtype=np.uint16)
    arr.tofile(bin_file)

    block_size = 4
    dataset = GPTDataset(data_path=bin_file, block_size=block_size)

    # Setup DataLoader
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over a batch
    for x_batch, y_batch in loader:
        assert x_batch.shape == (batch_size, block_size)
        assert y_batch.shape == (batch_size, block_size)
        assert x_batch.dtype == torch.long
        assert y_batch.dtype == torch.long
        break
