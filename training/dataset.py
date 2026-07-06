"""PyTorch Dataset and preprocessing utilities for Mini GPT dataset loading.

This module provides:
1. GPTDataset: A memory-mapped PyTorch Dataset wrapper for sequence input/target creation.
2. preprocess_data: A function to tokenize raw text and save it as preprocessed binaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from model.config import GPTConfig
from tokenizer.tokenizer import BaseTokenizer


class GPTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-efficient PyTorch Dataset loading sequence segments from a binary file."""

    def __init__(self, data_path: str | Path, block_size: int) -> None:
        """Initialize the GPTDataset.

        Args:
            data_path: Path to the preprocessed binary (.bin) file.
            block_size: Max context sequence length.
        """
        self.data_path = Path(data_path)
        self.block_size = block_size

        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed dataset binary not found at: {self.data_path}")

        # Use memory mapping (memmap) to avoid loading the entire dataset into RAM
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")

        if len(self.data) < block_size + 1:
            raise ValueError(
                f"Dataset length {len(self.data)} is too small for block_size {block_size} + 1."
            )

    def __len__(self) -> int:
        # Number of non-overlapping sequence blocks of length block_size
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate starting position for non-overlapping block
        start_pos = idx * self.block_size
        
        # Extract inputs (x) and targets (y), shifting y by 1 token
        x_np = np.array(self.data[start_pos : start_pos + self.block_size], dtype=np.int64)
        y_np = np.array(self.data[start_pos + 1 : start_pos + self.block_size + 1], dtype=np.int64)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x, y


def preprocess_data(config: GPTConfig, tokenizer: BaseTokenizer) -> None:
    """Read a raw text corpus, tokenize it, split it, and save train/val binaries.

    Args:
        config: GPTConfig containing data paths and split ratio.
        tokenizer: Trained tokenizer instance.
    """
    raw_path = Path(config.data.raw_data_path)
    processed_dir = Path(config.data.processed_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    # Create target directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw data from {raw_path}...")
    with open(raw_path, encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing text ({len(text)} characters)...")
    ids = tokenizer.encode(text)
    print(f"Total tokens generated: {len(ids)}")

    # Split into train and val segments
    split_idx = int(len(ids) * config.data.train_split)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    print(f"Splitting data (train split: {config.data.train_split}):")
    print(f"  - Train tokens: {len(train_ids)}")
    print(f"  - Val tokens: {len(val_ids)}")

    # Convert to numpy uint16 arrays for space-efficient storage
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    # Save to binary files
    train_path = processed_dir / "train.bin"
    val_path = processed_dir / "val.bin"

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print("Successfully saved preprocessed binaries to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")


if __name__ == "__main__":
    import argparse
    from generate import load_tokenizer
    
    parser = argparse.ArgumentParser(description="Preprocess raw text dataset into binary formats.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="data/tokenizer",
        help="Directory containing the trained tokenizer.",
    )
    args = parser.parse_args()

    if args.config:
        config = GPTConfig.from_yaml(args.config)
    else:
        config = GPTConfig()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    preprocess_data(config, tokenizer)
