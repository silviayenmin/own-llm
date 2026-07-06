"""Training engine for ownllm.

Coordinates model optimization, learning rate updates, periodic validation
evaluation, TensorBoard metrics logging, and model checkpointing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.config import GPTConfig, TrainingConfig


class Trainer:
    """Core training loop engine for Mini GPT models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler_fn: Callable[[int, TrainingConfig], float] | None = None,
        full_config: GPTConfig | None = None,
        output_name: str = "best_model.pt",
    ) -> None:
        """Initialize the trainer.

        Args:
            model: GPT model to train.
            optimizer: Configured optimizer.
            config: TrainingConfig with epochs, device, logging intervals, etc.
            train_loader: DataLoader containing the training data.
            val_loader: DataLoader containing the validation data.
            scheduler_fn: Optional learning rate scheduling function.
            output_name: Filename for the saved checkpoint.
        """
        self.model = model
        self.output_name = output_name
        self.optimizer = optimizer
        self.config = config
        self.full_config = full_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler_fn = scheduler_fn
        self.device = torch.device(config.device)

        # Move model to target computation device
        self.model.to(self.device)

        self.step = 0
        self.best_val_loss = float("inf")

        # Initialize TensorBoard metrics logger
        self.writer = SummaryWriter(log_dir=os.path.join(config.checkpoint_dir, "logs"))

    def train(self) -> None:
        """Run the complete training execution for config.epochs epochs."""
        self.model.train()
        print(f"Starting training on device: {self.device} for {self.config.epochs} epochs.")

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.config.epochs} ---")
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # 1) Adjust learning rate if scheduler is configured
                if self.scheduler_fn is not None:
                    lr = self.scheduler_fn(self.step, self.config)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    # Fetch from optimizer directly if no scheduler is used
                    lr = self.optimizer.param_groups[0]["lr"]

                # 2) Model forward pass and loss extraction
                self.optimizer.zero_grad(set_to_none=True)
                logits, loss = self.model(x, y)
                loss = cast(torch.Tensor, loss)

                # 3) Backward optimization
                loss.backward()

                # 4) Gradient norm clipping
                if self.config.grad_clip > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    ).item()
                else:
                    # Compute norm without clipping for logging metrics
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm**0.5

                # 5) Weight updates
                self.optimizer.step()

                # Logging to TensorBoard
                self.writer.add_scalar("loss/train", loss.item(), self.step)
                self.writer.add_scalar("lr", lr, self.step)
                self.writer.add_scalar("grad_norm", grad_norm, self.step)

                # Periodically log progress to standard output
                if self.step % self.config.log_interval == 0:
                    print(
                        f"Step {self.step:6d} | Batch {batch_idx:4d} | "
                        f"Train Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                        f"Grad Norm: {grad_norm:.4f}"
                    )

                # Periodically execute validation evaluation steps
                if self.step > 0 and self.step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.writer.add_scalar("loss/val", val_loss, self.step)
                    print(
                        f"[Eval Step] Step {self.step} | Validation Loss: {val_loss:.4f} "
                        f"(Best: {self.best_val_loss:.4f})"
                    )

                    # Save checkpoint if validation loss improves
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(val_loss)

                self.step += 1

        # Run one final evaluation and save checkpoint at the end of training
        final_val_loss = self.evaluate()
        self.writer.add_scalar("loss/val", final_val_loss, self.step)
        print(f"\nTraining completed. Final Validation Loss: {final_val_loss:.4f}")
        if final_val_loss < self.best_val_loss:
            self.best_val_loss = final_val_loss
            self.save_checkpoint(final_val_loss)

        self.writer.close()

    def evaluate(self) -> float:
        """Evaluate the model loss on validation dataset.

        Returns:
            The average validation loss across eval_iters batches.
        """
        self.model.eval()
        losses = []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= self.config.eval_iters:
                    break
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                loss = cast(torch.Tensor, loss)
                losses.append(loss.item())

        self.model.train()
        if len(losses) == 0:
            return 0.0
        return sum(losses) / len(losses)

    def save_checkpoint(self, val_loss: float) -> None:
        """Serialize current model and optimizer states to disk.

        Args:
            val_loss: Validation loss associated with the checkpoint.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, self.output_name)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "val_loss": val_loss,
            "config": self.full_config.model_dump()
            if self.full_config is not None
            else self.config.model_dump(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved new best checkpoint to {checkpoint_path}")
