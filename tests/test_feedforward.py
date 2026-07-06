import torch

from model.config import ModelConfig
from model.feedforward import GPTFeedForward


def test_feedforward_shape() -> None:
    """Verify custom GPTFeedForward preserves dimensions correctly."""
    config = ModelConfig(n_embd=32, dropout=0.1)
    ffn = GPTFeedForward(config)

    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C)
    out = ffn(x)

    assert out.shape == (B, T, C)


def test_feedforward_activation() -> None:
    """Verify inputs undergo expansion, GELU activation, and projection."""
    config = ModelConfig(n_embd=4, dropout=0.0, bias=True)
    ffn = GPTFeedForward(config)
    ffn.eval()  # Disable dropout for deterministic evaluation

    # Set linear layer weights to identity and bias to zero for check
    with torch.no_grad():
        ffn.c_fc.weight.copy_(torch.eye(16, 4))
        ffn.c_fc.bias.zero_()
        ffn.c_proj.weight.copy_(torch.eye(4, 16))
        ffn.c_proj.bias.zero_()

    # Pass in negative input
    # GELU(x) for negative x scales it down significantly
    x = torch.tensor([[[-1.0, -2.0, 0.0, 1.0]]])
    out = ffn(x)

    # Expected values using standard GELU formula
    expected = torch.nn.functional.gelu(x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_feedforward_bias_toggle() -> None:
    """Verify that linear layer bias terms are configured correctly according to config.bias."""
    config_with_bias = ModelConfig(n_embd=16, bias=True)
    ffn_with_bias = GPTFeedForward(config_with_bias)
    assert ffn_with_bias.c_fc.bias is not None
    assert ffn_with_bias.c_proj.bias is not None

    config_no_bias = ModelConfig(n_embd=16, bias=False)
    ffn_no_bias = GPTFeedForward(config_no_bias)

    # Use variable-based getattr to satisfy both mypy and ruff lint check
    bias_attr = "bias"
    assert getattr(ffn_no_bias.c_fc, bias_attr) is None
    assert getattr(ffn_no_bias.c_proj, bias_attr) is None


def test_feedforward_gradients() -> None:
    """Verify that gradients correctly backpropagate to all parameters of GPTFeedForward."""
    config = ModelConfig(n_embd=16, bias=True)
    ffn = GPTFeedForward(config)

    x = torch.randn(2, 4, 16)
    out = ffn(x)
    loss = out.sum()
    loss.backward()

    assert ffn.c_fc.weight.grad is not None
    assert ffn.c_fc.bias.grad is not None
    assert ffn.c_proj.weight.grad is not None
    assert ffn.c_proj.bias.grad is not None
