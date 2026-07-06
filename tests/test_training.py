import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from model.config import GPTConfig, TrainingConfig
from model.gpt import GPT
from training.dataset import GPTDataset
from training.loss import calculate_loss
from training.optimizer import get_optimizer
from training.scheduler import get_learning_rate
from training.trainer import Trainer


def test_calculate_loss() -> None:
    """Verify that calculate_loss wraps cross-entropy loss correctly."""
    B, T, vocab_size = 2, 4, 10
    logits = torch.randn(B, T, vocab_size)
    targets = torch.randint(0, vocab_size, (B, T))

    loss_1 = calculate_loss(logits, targets)
    loss_2 = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), targets.view(-1), ignore_index=-1
    )

    assert torch.allclose(loss_1, loss_2)


def test_optimizer_parameter_groups() -> None:
    """Verify that get_optimizer separates weights and biases/LayerNorms for weight decay."""
    config = GPTConfig()
    config.model.n_embd = 16
    config.model.n_head = 2
    config.model.n_layer = 1
    config.model.vocab_size = 50
    model = GPT(config)

    training_config = TrainingConfig(weight_decay=0.1)
    optimizer = get_optimizer(model, training_config)

    # Check optimizer groups
    assert len(optimizer.param_groups) == 2

    decay_group = optimizer.param_groups[0]
    nodecay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert nodecay_group["weight_decay"] == 0.0

    # Ensure parameter sets partition model parameters cleanly
    decay_params = set(decay_group["params"])
    nodecay_params = set(nodecay_group["params"])

    all_model_params = {p for p in model.parameters() if p.requires_grad}

    assert decay_params.union(nodecay_params) == all_model_params
    assert len(decay_params.intersection(nodecay_params)) == 0

    # Check specific parameter placement
    # wte weight should decay (2D parameter)
    assert model.transformer.wte.embedding.weight in decay_params

    # LayerNorm weights and linear biases should NOT decay
    assert model.transformer.ln_f.weight in nodecay_params
    if model.transformer.ln_f.bias is not None:
        assert model.transformer.ln_f.bias in nodecay_params


def test_scheduler_learning_rate() -> None:
    """Verify get_learning_rate warmup and cosine decay values at key steps."""
    config = TrainingConfig(
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=10,
        lr_decay_iters=100,
    )

    # 1) Start of warmup
    assert get_learning_rate(0, config) == 0.0

    # 2) Middle of warmup
    assert get_learning_rate(5, config) == 5e-4

    # 3) End of warmup (peak learning rate)
    assert pytest.approx(get_learning_rate(10, config)) == 1e-3

    # 4) Post decay threshold (floor learning rate)
    assert get_learning_rate(101, config) == 1e-4

    # 5) Cosine decay mid-point
    # (10 + 100) / 2 = 55
    # decay ratio = (55 - 10) / (100 - 10) = 45 / 90 = 0.5
    # coeff = 0.5 * (1 + cos(pi/2)) = 0.5 * (1 + 0) = 0.5
    # expected lr = min_lr + 0.5 * (lr - min_lr) = 1e-4 + 0.5 * (9e-4) = 5.5e-4
    assert pytest.approx(get_learning_rate(55, config)) == 5.5e-4


def test_trainer_mini_run() -> None:
    """Run a micro training run using temporary paths to verify optimizer, trainer and checkpoints."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock preprocessed binary file with some integers
        mock_data = np.arange(20, dtype=np.int64)
        mock_bin_path = os.path.join(temp_dir, "data.bin")
        mock_memmap = np.memmap(mock_bin_path, dtype=np.int64, mode="w+", shape=(20,))
        mock_memmap[:] = mock_data[:]
        mock_memmap.flush()

        config = GPTConfig()
        config.model.n_embd = 16
        config.model.n_head = 2
        config.model.n_layer = 1
        config.model.vocab_size = 50
        config.model.block_size = 4

        config.training.batch_size = 2
        config.training.epochs = 2
        config.training.log_interval = 1
        config.training.eval_interval = 2
        config.training.eval_iters = 2
        config.training.checkpoint_dir = temp_dir
        config.training.device = "cpu"
        config.training.learning_rate = 1e-3

        model = GPT(config)
        optimizer = get_optimizer(model, config.training)

        train_dataset = GPTDataset(mock_bin_path, config.model.block_size)
        val_dataset = GPTDataset(mock_bin_path, config.model.block_size)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config.training,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler_fn=get_learning_rate,
            full_config=config,
        )

        # Confirm training step runs without error
        trainer.train()

        # Check if checkpoints and TensorBoard log directory were created
        checkpoint_file = os.path.join(temp_dir, "best_model.pt")
        assert os.path.exists(checkpoint_file)

        # Load check to verify structure
        checkpoint = torch.load(checkpoint_file)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "step" in checkpoint
        assert checkpoint["val_loss"] is not None

        # Verify validation function computes losses correctly
        val_loss = trainer.evaluate()
        assert isinstance(val_loss, float)
        assert val_loss >= 0.0

        # Close memmap references before cleanup to prevent WinError 32
        del mock_memmap
        del train_dataset
        del val_dataset
        del train_loader
        del val_loader
        del trainer
        import gc
        gc.collect()

    finally:
        shutil.rmtree(temp_dir)
