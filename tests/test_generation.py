import os
import shutil
import sys
import tempfile
from unittest.mock import patch

import torch

from generate import load_tokenizer, main
from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.tokenizer import CharTokenizer


def test_gpt_generate_shapes() -> None:
    """Verify that model.generate appends the requested number of tokens."""
    config = GPTConfig()
    config.model.vocab_size = 50
    config.model.n_embd = 16
    config.model.n_head = 2
    config.model.n_layer = 1
    config.model.block_size = 16
    model = GPT(config)
    model.eval()

    B, T = 2, 4
    idx = torch.randint(0, 50, (B, T))

    max_new_tokens = 5
    out = model.generate(idx, max_new_tokens=max_new_tokens)

    # Output shape should be (B, T + max_new_tokens)
    assert out.shape == (B, T + max_new_tokens)


def test_gpt_generate_top_k() -> None:
    """Verify top-k sampling restricts the next token choices correctly."""
    config = GPTConfig()
    config.model.vocab_size = 50
    config.model.n_embd = 16
    config.model.n_head = 2
    config.model.n_layer = 1
    config.model.block_size = 16
    model = GPT(config)
    model.eval()

    # Create prompt input
    idx = torch.randint(0, 50, (1, 4))

    # Test top-k = 1: next generated tokens should be fully deterministic
    # We will generate 3 tokens twice and verify they are identical
    out_1 = model.generate(idx, max_new_tokens=3, top_k=1)
    out_2 = model.generate(idx, max_new_tokens=3, top_k=1)

    assert torch.equal(out_1, out_2)


def test_load_tokenizer_routing() -> None:
    """Verify that load_tokenizer correctly routes to CharTokenizer or BPETokenizer based on config."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a mock CharTokenizer and save it
        chars = ["a", "b", "c"]
        tokenizer = CharTokenizer(chars=chars, special_tokens=["<|endoftext|>"])
        tokenizer.save(temp_dir)

        # Load it back via routing utility
        loaded = load_tokenizer(temp_dir)
        assert isinstance(loaded, CharTokenizer)
        assert loaded.vocab_size == tokenizer.vocab_size
    finally:
        shutil.rmtree(temp_dir)


def test_generation_cli_end_to_end() -> None:
    """Verify that generate.py CLI main parses arguments, loads checkpoints and generates text."""
    temp_dir = tempfile.mkdtemp()
    try:
        # 1) Save a mock CharTokenizer
        chars = ["a", "b", "c", "d", " "]
        tokenizer = CharTokenizer(chars=chars, special_tokens=["<|endoftext|>"])
        tokenizer_dir = os.path.join(temp_dir, "tokenizer")
        tokenizer.save(tokenizer_dir)

        # 2) Save a mock model checkpoint
        config = GPTConfig()
        config.model.vocab_size = tokenizer.vocab_size
        config.model.n_embd = 16
        config.model.n_head = 2
        config.model.n_layer = 1
        config.model.block_size = 8
        model = GPT(config)

        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "step": 0,
            "val_loss": 0.0,
            "config": config.model_dump(),
        }
        torch.save(checkpoint, checkpoint_path)

        # 3) Patch CLI args and run main
        test_args = [
            "generate.py",
            "--checkpoint", checkpoint_path,
            "--prompt", "abc",
            "--max_new_tokens", "5",
            "--device", "cpu",
            "--tokenizer_dir", tokenizer_dir,
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Verify executing with empty prompt (defaults to special token) works too
        test_args_empty_prompt = [
            "generate.py",
            "--checkpoint", checkpoint_path,
            "--max_new_tokens", "2",
            "--device", "cpu",
            "--tokenizer_dir", tokenizer_dir,
        ]
        with patch.object(sys, "argv", test_args_empty_prompt):
            main()

    finally:
        shutil.rmtree(temp_dir)
