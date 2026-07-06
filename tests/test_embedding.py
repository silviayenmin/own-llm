import pytest
import torch

from model.config import ModelConfig
from model.embedding import GPTPositionalEmbedding, GPTTokenEmbedding


def test_token_embedding_forward() -> None:
    """Verify GPTTokenEmbedding returns the correct output shapes and lookup mappings."""
    config = ModelConfig(vocab_size=100, n_embd=32)
    token_emb = GPTTokenEmbedding(config)

    # Batch of size 2, sequence length 5
    B, T = 2, 5
    x = torch.randint(0, config.vocab_size, (B, T))
    out = token_emb(x)

    # Verify shape
    assert out.shape == (B, T, config.n_embd)

    # Verify manual mapping match
    # Weight shape should be (vocab_size, n_embd)
    assert token_emb.embedding.weight.shape == (config.vocab_size, config.n_embd)
    first_token_id = int(x[0, 0].item())
    expected_vector = token_emb.embedding.weight[first_token_id]
    assert torch.allclose(out[0, 0], expected_vector)


def test_positional_embedding_forward() -> None:
    """Verify GPTPositionalEmbedding returns the correct shape for both integer and tensor inputs."""
    config = ModelConfig(block_size=10, n_embd=16)
    pos_emb = GPTPositionalEmbedding(config)

    # Test with integer input
    T_int = 5
    out_int = pos_emb(T_int)
    assert out_int.shape == (T_int, config.n_embd)

    # Test with tensor input of shape (B, T)
    B, T_tensor = 2, 6
    x = torch.randint(0, 100, (B, T_tensor))
    out_tensor = pos_emb(x)
    assert out_tensor.shape == (1, T_tensor, config.n_embd)

    # Verify value match: positional embedding index 0 matches the first position
    assert torch.allclose(out_tensor[0, 0], pos_emb.embedding.weight[0])


def test_positional_embedding_invalid() -> None:
    """Verify GPTPositionalEmbedding raises ValueError for negative or zero length inputs."""
    config = ModelConfig(block_size=10, n_embd=16)
    pos_emb = GPTPositionalEmbedding(config)

    with pytest.raises(ValueError):
        pos_emb(0)

    with pytest.raises(ValueError):
        pos_emb(-5)


def test_device_matching() -> None:
    """Verify GPTPositionalEmbedding device matching behavior for GPU testing simulation."""
    config = ModelConfig(block_size=10, n_embd=16)
    pos_emb = GPTPositionalEmbedding(config)

    # Create tensor on a specific device (fallback to CPU, but use mock device if possible)
    device = torch.device("cpu")
    x = torch.randint(0, 10, (1, 4), device=device)
    out = pos_emb(x)
    assert out.device == device


def test_embedding_gradients() -> None:
    """Verify that both Token and Positional Embeddings accumulate gradients during backward pass."""
    config = ModelConfig(vocab_size=50, block_size=10, n_embd=16)
    token_emb = GPTTokenEmbedding(config)
    pos_emb = GPTPositionalEmbedding(config)

    # Forward pass
    x = torch.randint(0, config.vocab_size, (2, 4))
    tok_out = token_emb(x)
    pos_out = pos_emb(x)

    # Combine embeddings (as in GPT)
    combined = tok_out + pos_out

    # Compute mock loss and backward
    loss = combined.sum()
    loss.backward()

    # Verify gradients exist
    assert token_emb.embedding.weight.grad is not None
    assert pos_emb.embedding.weight.grad is not None
    assert token_emb.embedding.weight.grad.shape == token_emb.embedding.weight.shape
    assert pos_emb.embedding.weight.grad.shape == pos_emb.embedding.weight.shape
