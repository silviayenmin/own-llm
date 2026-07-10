"""Script to convert a JSON file of Q&A pairs into a binary dataset for instruction fine-tuning."""

import argparse
import json
import numpy as np
from pathlib import Path

from model.config import GPTConfig
from generate import load_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Preprocess instruction dataset")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    args = parser.parse_args()

    config = GPTConfig.from_yaml(args.config) if args.config else GPTConfig()
    
    # Resolve tokenizer directory from config if provided
    tokenizer_dir = config.data.tokenizer_dir if args.config else "data/tokenizer"
    tokenizer = load_tokenizer(tokenizer_dir)
    
    import torch

    json_path = Path("data/instruct/qa_dataset.json")
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found!")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
        
    # Block size + 1 to account for x and y shifts
    seq_len = config.model.block_size + 1
    
    # Get the ID for the padding token
    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special=True)[0]

    all_encoded_pairs = []
    
    for item in qa_pairs:
        # Note: <|endoftext|> tells the model to stop generating
        formatted_text = f"Prompt: {item['prompt']}\nResponse: {item['response']}<|endoftext|>"
        
        # Tokenize the individual QA pair
        ids = tokenizer.encode(formatted_text, allowed_special=True)
        
        # Truncate if longer than seq_len
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            
        # Pad if shorter than seq_len
        if len(ids) < seq_len:
            ids = ids + [pad_token_id] * (seq_len - len(ids))
            
        all_encoded_pairs.append(ids)
        
    print(f"Formatted {len(qa_pairs)} Q&A pairs into padded sequences of length {seq_len}.")
    
    # Convert to PyTorch Tensor of shape (N, seq_len)
    dataset_tensor = torch.tensor(all_encoded_pairs, dtype=torch.long)
    
    # Save as instruct_train.pt
    out_path = Path(config.data.processed_dir) / "instruct_train.pt"
    torch.save(dataset_tensor, out_path)
    print(f"Saved padded tensor to {out_path}")
    
    # We will use the same tensor for validation just for simplicity since it's a tiny dataset
    val_path = Path(config.data.processed_dir) / "instruct_val.pt"
    torch.save(dataset_tensor, val_path)
    print(f"Saved validation tensor to {val_path}")

if __name__ == "__main__":
    main()
