"""Command Line Interface for text generation using a trained GPT model.

Loads model checkpoints, tokenizes the input prompt, runs autoregressive text
generation, and decodes the outputs to standard output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.tokenizer import BaseTokenizer, BPETokenizer, CharTokenizer


def load_tokenizer(directory: str | Path) -> BaseTokenizer:
    """Detect and load the correct tokenizer type (BPE or Char) from directory.

    Args:
        directory: Tokenizer directory containing tokenizer.json config.

    Returns:
        Loaded BaseTokenizer subclass instance.
    """
    path = Path(directory) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer configuration file not found at: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tok_type = data.get("type", "bpe")
    if tok_type == "bpe":
        return BPETokenizer.load(directory)
    elif tok_type == "char":
        return CharTokenizer.load(directory)
    else:
        raise ValueError(f"Unknown tokenizer type '{tok_type}' serialized in {path}")


def main() -> None:
    """Parse generation parameters, execute model generation, and print output."""
    parser = argparse.ArgumentParser(description="Generate text using a trained Mini GPT model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint file (.pt).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt string to seed text generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (lower = more deterministic, higher = more random).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional Top-K filtering parameter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Backend to run inference on ('cpu', 'cuda', 'mps'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="data/tokenizer",
        help="Directory containing the trained tokenizer.",
    )
    args = parser.parse_args()

    # Determine device target
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using target device backend: {device}")

    # Load model checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Reconstruct configuration and load weights
    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)

    # Force the config device to match current runtime device
    config.training.device = device

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = load_tokenizer(args.tokenizer_dir)

    prompt_text = args.prompt
    if not prompt_text:
        # Seed with endoftext special token if no prompt is provided
        prompt_text = "<|endoftext|>"

    print(f"Prompt sequence: '{prompt_text}'")
    encoded_ids = tokenizer.encode(prompt_text, allowed_special=True)
    x = torch.tensor(encoded_ids, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, T)

    # Generate sequence
    print(f"Generating up to {args.max_new_tokens} tokens...")
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # Decode and print generated text
    generated_ids = y[0].tolist()
    prompt_len = len(encoded_ids)
    pad_token_id = tokenizer.special_stoi.get("<|endoftext|>", 256)
    new_tokens = generated_ids[prompt_len:]
    if pad_token_id in new_tokens:
        first_pad_idx = new_tokens.index(pad_token_id)
        generated_ids = generated_ids[:prompt_len + first_pad_idx]
        
    output_text = tokenizer.decode(generated_ids)
    print("\n--- Generated Output ---")
    print(output_text)
    print("------------------------")


if __name__ == "__main__":
    main()
