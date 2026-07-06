"""Optimizer configuration utility for ownllm.

Configures AdamW optimizer with selective weight decay for 2D parameters
(weights of Linear/Embedding layers) and zero weight decay for 1D parameters
(biases, LayerNorm parameters).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import TrainingConfig


def get_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Configure the AdamW optimizer with selective weight decay.

    Separates parameters into decayable (weights of Linear and Embedding layers)
    and non-decayable (biases and LayerNorm scale/shift vectors) parameters.

    Args:
        model: PyTorch model whose parameters need optimizing.
        config: TrainingConfig containing learning rate, decay, and beta hyperparams.

    Returns:
        Configured AdamW optimizer.
    """
    # Filter for parameters requiring gradient updates
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = []
    nodecay_params = []

    for name, param in param_dict.items():
        # Any parameter that is a bias or is a 1D tensor (LayerNorm weights, biases) gets no decay
        if name.endswith(".bias") or (name.endswith(".weight") and len(param.shape) < 2):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    return optimizer
