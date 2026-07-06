# Training Directory

This directory implements the training pipeline for our Mini GPT model.

## Modules

- `dataset.py`: PyTorch `Dataset` and `DataLoader` wrapper for sequence input/target creation.
- `loss.py`: Custom or wrapped cross-entropy loss function.
- `optimizer.py`: AdamW / SGD optimizer setup matching GPT hyperparameters.
- `scheduler.py`: Learning rate scheduler with cosine decay and warmup.
- `trainer.py`: Comprehensive training loop engine managing epochs, evaluation steps, TensorBoard logging, and checkpointing.
- `train.py`: Main entry point script to load configs, initialize modules, and run the trainer.
