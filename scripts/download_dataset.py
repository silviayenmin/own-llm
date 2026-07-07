"""Script to download a huge dataset (TinyStories) for training the large model."""

import argparse
import os
from pathlib import Path
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download TinyStories dataset")
    parser.add_argument("--output_file", type=str, default="data/raw/huge_corpus.txt", help="Output text file")
    # TinyStories has a lot of rows (millions). 
    # For a 16GB GPU running 124M params, 2.5 million rows (about 500MB) is a massive dataset.
    parser.add_argument("--samples", type=int, default=2500000, help="Number of story samples to download")
    args = parser.parse_args()

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading 'roneneldan/TinyStories' from HuggingFace...")
    # 'train' split has about 2.11M rows
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    print(f"Writing first {args.samples} stories to {out_path}...")
    
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(min(args.samples, len(dataset))):
            text = dataset[i]["text"].strip()
            if text:
                f.write(text + "\n<|endoftext|>\n")
                count += 1
            if count >= args.samples:
                break
                
    print(f"Done! Saved {count} stories to {out_path}.")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
