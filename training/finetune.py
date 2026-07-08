"""Fine-tuning script to train the base GPT model on instruction data."""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from model.config import GPTConfig
from model.gpt import GPT
from training.dataset import InstructDataset
from training.optimizer import get_optimizer
from training.scheduler import get_learning_rate
from training.trainer import Trainer
from training.train import set_seed

def main():
    parser = argparse.ArgumentParser(description="Fine-tune custom Mini GPT model.")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    args = parser.parse_args()

    config = GPTConfig.from_yaml(args.config) if args.config else GPTConfig()
    set_seed(config.training.seed)

    # Base model checkpoint
    checkpoint_path = os.path.join(config.training.checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Base model checkpoint not found at {checkpoint_path}")

    # Instruction dataset
    train_bin = os.path.join(config.data.processed_dir, "instruct_train.pt")
    val_bin = os.path.join(config.data.processed_dir, "instruct_val.pt")

    if not os.path.exists(train_bin):
        raise FileNotFoundError("Instruction binary not found. Run finetune_dataset.py first.")

    print("Loading instruction dataset splits...")
    train_dataset = InstructDataset(train_bin)
    val_dataset = InstructDataset(val_bin)

    pin_memory = config.training.device.startswith("cuda")
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=pin_memory)

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"Loading base model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Init model and load weights
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Overwrite config for fine-tuning
    config.training.learning_rate = 5e-4  # Slightly higher LR
    if len(train_dataset) > 50:
        config.training.epochs = 5        # Larger dataset needs fewer epochs to generalize
    else:
        config.training.epochs = 100      # Memorize toy QA dataset
    config.training.warmup_iters = 10
    
    optimizer = get_optimizer(model, config.training)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler_fn=get_learning_rate,
        full_config=config,
        output_name="instruct_model.pt"
    )

    print("Starting fine-tuning...")
    trainer.train()

if __name__ == "__main__":
    main()
