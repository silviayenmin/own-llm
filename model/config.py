"""Configuration schemas for the Mini GPT model, training harness, and data processing.

This module defines structures using Pydantic for validation and serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration options for the transformer architecture."""

    vocab_size: int = Field(
        default=50304,
        description="Vocabulary size (padded to a multiple of 64 for execution efficiency).",
        gt=0,
    )
    block_size: int = Field(
        default=512,
        description="Maximum context window / sequence length.",
        gt=0,
    )
    n_layer: int = Field(
        default=6,
        description="Number of transformer block layers.",
        gt=0,
    )
    n_head: int = Field(
        default=6,
        description="Number of self-attention heads.",
        gt=0,
    )
    n_embd: int = Field(
        default=384,
        description="Dimensionality of the embedding and block layers.",
        gt=0,
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout probability applied within attention and FFN layers.",
        ge=0.0,
        le=1.0,
    )
    bias: bool = Field(
        default=False,
        description="Whether to use bias terms in linear mappings (GPT-2 uses True, NanoGPT uses False).",
    )


class TrainingConfig(BaseModel):
    """Hyperparameters and configuration settings for the training pipeline."""

    batch_size: int = Field(default=64, description="Training micro-batch size.", gt=0)
    learning_rate: float = Field(default=6.0e-4, description="Peak learning rate.", gt=0.0)
    weight_decay: float = Field(default=0.1, description="AdamW weight decay.", ge=0.0)
    beta1: float = Field(default=0.9, description="Adam optimizer beta1.", ge=0.0, le=1.0)
    beta2: float = Field(default=0.95, description="Adam optimizer beta2.", ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, description="Gradient clipping scale.", ge=0.0)
    epochs: int = Field(default=10, description="Total training epochs.", gt=0)
    warmup_iters: int = Field(default=1000, description="Learning rate warmup steps.", ge=0)
    lr_decay_iters: int = Field(default=50000, description="Decay window in step count.", gt=0)
    min_lr: float = Field(default=6.0e-5, description="Minimum learning rate floor.", ge=0.0)
    eval_interval: int = Field(default=500, description="Frequency of model evaluation.", gt=0)
    eval_iters: int = Field(default=200, description="Iteration batches used during evaluation.", gt=0)
    log_interval: int = Field(default=10, description="Metrics print frequency.", gt=0)
    checkpoint_dir: str = Field(default="experiments/checkpoints", description="Directory to save weights.")
    device: str = Field(default="cuda", description="Computation backend ('cuda', 'cpu', 'mps').")
    seed: int = Field(default=1337, description="Random seed for reproducibility.")


class DataConfig(BaseModel):
    """Configuration paths and preprocessing configuration."""

    raw_data_path: str = Field(default="data/raw/input.txt", description="Path to source text file.")
    processed_dir: str = Field(default="data/processed", description="Target directory for processed binaries.")
    tokenizer_dir: str = Field(default="data/tokenizer", description="Storage directory for trained tokenizer.")
    train_split: float = Field(
        default=0.9,
        description="Ratio of data reserved for the training split.",
        gt=0.0,
        lt=1.0,
    )


class GPTConfig(BaseModel):
    """Unified configuration compiling all sub-configs for Mini GPT execution."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GPTConfig:
        """Load and parse configuration settings from a YAML file.

        Args:
            path: Target YAML configuration filepath.

        Returns:
            An instance of GPTConfig containing validated options.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls(**raw_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the validated configuration object back to a standard Python dictionary.

        Returns:
            A nested dictionary of the configs.
        """
        return self.model_dump()
