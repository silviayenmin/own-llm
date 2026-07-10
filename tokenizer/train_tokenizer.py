"""Script to train a custom tokenizer on a text corpus.

Supports training either a Byte-Pair Encoding (BPE) tokenizer or a character-level tokenizer,
using settings defined in the project configuration or passed as arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model.config import GPTConfig
from tokenizer.tokenizer import BaseTokenizer, BPETokenizer, CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom tokenizer for Mini GPT.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["bpe", "char"],
        default=None,
        help="Tokenizer type to train ('bpe' or 'char'). Defaults to config or 'bpe'.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Target vocabulary size (BPE only). Defaults to config value.",
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default=None,
        help="Path to the input training corpus. Defaults to config value.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Directory to save the trained tokenizer files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose training logs (useful for BPE merges).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = GPTConfig.from_yaml(args.config)
    else:
        print("Using default configuration settings...")
        config = GPTConfig()

    # Resolve arguments, falling back to config
    tokenizer_type = args.type or "bpe"
    vocab_size = args.vocab_size or config.model.vocab_size
    raw_data_path = Path(args.raw_data_path or config.data.raw_data_path)
    tokenizer_dir = Path(args.tokenizer_dir or config.data.tokenizer_dir)

    print("Training parameters:")
    print(f"  - Tokenizer Type: {tokenizer_type}")
    print(f"  - Input Data Path: {raw_data_path}")
    print(f"  - Save Directory: {tokenizer_dir}")
    if tokenizer_type == "bpe":
        print(f"  - Target Vocabulary Size: {vocab_size}")

    # Read training text
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {raw_data_path}. "
            "Please place your text corpus there or specify another path using --raw_data_path."
        )

    with open(raw_data_path, encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded training text ({len(text)} characters / {len(text.encode('utf-8'))} bytes).")

    # Define standard special tokens
    special_tokens = ["<|endoftext|>"]

    tokenizer: BaseTokenizer

    if tokenizer_type == "char":
        print("Extracting unique characters for Character Tokenizer...")
        unique_chars = sorted(set(text))
        print(f"Found {len(unique_chars)} unique characters.")

        tokenizer = CharTokenizer(chars=unique_chars, special_tokens=special_tokens)
        print("Character Tokenizer created.")

    elif tokenizer_type == "bpe":
        # Limit text to 100KB for speed and memory stability in pure Python BPE training
        if len(text) > 100 * 1024:
            print("Limiting BPE training text to first 100KB to prevent memory/GC crashes on Windows...")
            text = text[:100 * 1024]

        print("Training Byte-Pair Encoding (BPE) Tokenizer from scratch...")
        bpe_tokenizer = BPETokenizer(special_tokens=special_tokens)

        # Guard against vocab size request smaller than base vocab + special tokens
        min_allowed_vocab = 256 + len(special_tokens)
        if vocab_size <= min_allowed_vocab:
            print(f"Warning: Requested vocab_size {vocab_size} is too small. "
                  f"Setting vocab_size to {min_allowed_vocab + 100} to allow merges.")
            vocab_size = min_allowed_vocab + 100

        import gc
        gc.disable()
        try:
            bpe_tokenizer.train(text, vocab_size=vocab_size, verbose=args.verbose)
        finally:
            gc.enable()
            
        tokenizer = bpe_tokenizer
        print("BPE Tokenizer training completed.")
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    # Save tokenizer
    tokenizer.save(tokenizer_dir)
    print(f"Successfully saved trained tokenizer to: {tokenizer_dir}")
    print(f"Final Vocabulary Size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
