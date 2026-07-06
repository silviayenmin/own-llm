"""Main training execution script for Mini GPT models.

Loads target configurations, initializes tokenizer, preprocessed datasets,
and executes the training loop.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.config import GPTConfig
from model.gpt import GPT
from training.dataset import GPTDataset
from training.optimizer import get_optimizer
from training.scheduler import get_learning_rate
from training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Load configuration and run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train custom Mini GPT model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    # Load unified configurations
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = GPTConfig.from_yaml(args.config)
    else:
        print("Using default model configurations.")
        config = GPTConfig()

    set_seed(config.training.seed)

    # Establish processed binary paths
    train_bin = os.path.join(config.data.processed_dir, "train.bin")
    val_bin = os.path.join(config.data.processed_dir, "val.bin")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(
            f"Preprocessed dataset binary files not found in: {config.data.processed_dir}\n"
            f"Please run preprocess steps or training/dataset.py first."
        )

    # Load datasets
    print("Loading preprocessed dataset splits...")
    train_dataset = GPTDataset(train_bin, config.model.block_size)
    val_dataset = GPTDataset(val_bin, config.model.block_size)

    # Configure DataLoaders
    # Pin memory speeds up host-to-device transfers on CUDA devices
    pin_memory = config.training.device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    print(f"Initializing GPT model containing {config.model.n_layer} layers...")
    model = GPT(config)

    print("Configuring optimizer and learning rate scheduler...")
    optimizer = get_optimizer(model, config.training)

    # Assemble trainer engine
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler_fn=get_learning_rate,
        full_config=config,
    )

    # Run the training loop
    trainer.train()


if __name__ == "__main__":
    main()
