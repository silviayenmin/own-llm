# Data Directory

This directory manages all raw text data, preprocessed training splits, and tokenizer artifacts.

## Subdirectories

- `raw/`: Stores raw input corpora (e.g., `.txt` files) before preprocessing. These files are read-only.
- `processed/`: Stores preprocessed tokenized datasets (e.g., numpy arrays, bin/idx files) ready to be loaded by PyTorch `Dataset` classes for training.
- `tokenizer/`: Stores trained tokenizer model files, vocabulary configs, and token frequencies.
