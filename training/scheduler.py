"""Learning rate scheduler utility for ownllm.

Computes learning rate for each training step using linear warmup followed by
cosine decay down to a minimum learning rate floor.
"""

from __future__ import annotations

import math

from model.config import TrainingConfig


def get_learning_rate(step: int, config: TrainingConfig) -> float:
    """Compute the target learning rate for the given training step.

    Applies linear warmup over warmup_iters steps, followed by a cosine decay
    curve down to min_lr over the remaining steps up to lr_decay_iters.

    Args:
        step: Current iteration step count (0-indexed).
        config: TrainingConfig containing scheduling hyperparameters.

    Returns:
        The target learning rate for the step.
    """
    # 1) Linear warmup phase
    if step < config.warmup_iters:
        return config.learning_rate * step / max(1, config.warmup_iters)

    # 2) Post-decay phase (constant min_lr floor)
    if step > config.lr_decay_iters:
        return config.min_lr

    # 3) Cosine decay phase
    decay_ratio = (step - config.warmup_iters) / max(
        1, config.lr_decay_iters - config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges from 1.0 to 0.0

    return config.min_lr + coeff * (config.learning_rate - config.min_lr)
