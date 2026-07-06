import pytest
import torch

from model.attention import GPTSelfAttention
from model.config import ModelConfig


def test_attention_shapes() -> None:
    """Verify that GPTSelfAttention preserves sequence dimension shapes correctly."""
    config = ModelConfig(n_embd=32, n_head=4, block_size=16)
    attn = GPTSelfAttention(config)

    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C)
    out = attn(x)

    assert out.shape == (B, T, C)


def test_attention_causal_masking() -> None:
    """Verify that future context tokens do not leak / affect present tokens (causality check)."""
    config = ModelConfig(n_embd=16, n_head=2, block_size=8, dropout=0.0)
    attn = GPTSelfAttention(config)
    attn.eval()  # Disable dropout for deterministic evaluation

    B, T, C = 1, 6, 16
    x1 = torch.randn(B, T, C)

    # Create x2 identical to x1 up to index 3, but completely different at index 4 and 5
    x2 = x1.clone()
    x2[0, 4:, :] = torch.randn(B, 2, C)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Verify that outputs are identical up to index 3 (no future leakage)
    assert torch.allclose(out1[0, :4, :], out2[0, :4, :], atol=1e-6)

    # Outputs at index 4 and 5 should differ due to the changed input suffix
    assert not torch.allclose(out1[0, 4:, :], out2[0, 4:, :], atol=1e-6)


def test_attention_divisibility_validation() -> None:
    """Verify that GPTSelfAttention raises ValueError if n_embd is not divisible by n_head."""
    config = ModelConfig(n_embd=33, n_head=6)
    with pytest.raises(ValueError):
        GPTSelfAttention(config)


def test_attention_bias_toggle() -> None:
    """Verify that the learnable bias terms are configured correctly according to config.bias."""
    config_with_bias = ModelConfig(n_embd=16, n_head=2, bias=True)
    attn_with_bias = GPTSelfAttention(config_with_bias)
    assert attn_with_bias.c_attn.bias is not None
    assert attn_with_bias.c_proj.bias is not None

    config_no_bias = ModelConfig(n_embd=16, n_head=2, bias=False)
    attn_no_bias = GPTSelfAttention(config_no_bias)
    bias_attr = "bias"
    assert getattr(attn_no_bias.c_attn, bias_attr) is None
    assert getattr(attn_no_bias.c_proj, bias_attr) is None


def test_attention_gradients() -> None:
    """Verify that gradients correctly backpropagate to all parameters of GPTSelfAttention."""
    config = ModelConfig(n_embd=16, n_head=2, bias=True)
    attn = GPTSelfAttention(config)

    x = torch.randn(2, 4, 16)
    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.c_attn.weight.grad is not None
    assert attn.c_attn.bias.grad is not None
    assert attn.c_proj.weight.grad is not None
    assert attn.c_proj.bias.grad is not None
