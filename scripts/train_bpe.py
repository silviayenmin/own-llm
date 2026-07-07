"""Script to train a custom BPE tokenizer on a huge corpus chunk."""

import argparse
import time
from pathlib import Path

from tokenizer.tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer")
    parser.add_argument("--input_file", type=str, default="data/raw/huge_corpus.txt", help="Path to raw corpus")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Target vocabulary size")
    parser.add_argument("--chunk_size_mb", type=float, default=5.0, help="How many MBs of data to use for training (to avoid pure python bottlenecks)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found! Please download the dataset first.")

    # Read a chunk of the text
    bytes_to_read = int(args.chunk_size_mb * 1024 * 1024)
    print(f"Reading up to {args.chunk_size_mb} MB from {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read(bytes_to_read)

    print(f"Read {len(text)} characters.")
    
    print(f"Training BPE Tokenizer with target vocab size {args.vocab_size}...")
    print("WARNING: Pure Python BPE training is O(N^2) and may take 5-15 minutes depending on chunk size.")
    
    tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])
    
    start = time.time()
    tokenizer.train(text, vocab_size=args.vocab_size, verbose=True)
    end = time.time()
    
    print(f"Training completed in {end - start:.2f} seconds.")
    
    out_dir = Path("data/tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir)
    print(f"Saved tokenizer to {out_dir}/tokenizer.json")

if __name__ == "__main__":
    main()
