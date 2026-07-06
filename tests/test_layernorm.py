import torch
import torch.nn as nn

from model.layernorm import GPTLayerNorm


def test_layernorm_shape_and_correctness() -> None:
    """Verify custom GPTLayerNorm returns the expected shape and mathematically correct statistics."""
    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    # Initialize layernorm with bias=False (only weights applied)
    ln = GPTLayerNorm(ndim=C, bias=False, eps=1e-5)

    # Temporarily set weights to 1.0 (so no scaling affects correctness check)
    with torch.no_grad():
        ln.weight.fill_(1.0)

    out = ln(x)
    assert out.shape == (B, T, C)

    # Mean along feature dimension should be very close to 0
    mean = out.mean(-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)

    # Variance along feature dimension should be very close to 1
    var = out.var(-1, unbiased=False)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_layernorm_equivalence_to_pytorch() -> None:
    """Verify custom GPTLayerNorm yields identical values to PyTorch's native nn.LayerNorm."""
    B, T, C = 2, 4, 16
    x = torch.randn(B, T, C)

    eps = 1e-6
    custom_ln = GPTLayerNorm(ndim=C, bias=True, eps=eps)
    pytorch_ln = nn.LayerNorm(normalized_shape=C, eps=eps)

    # Initialize weights and biases to match exactly
    with torch.no_grad():
        custom_ln.weight.copy_(pytorch_ln.weight)
        if custom_ln.bias is not None and pytorch_ln.bias is not None:
            custom_ln.bias.copy_(pytorch_ln.bias)

    custom_out = custom_ln(x)
    pytorch_out = pytorch_ln(x)

    assert torch.allclose(custom_out, pytorch_out, atol=1e-6)


def test_layernorm_bias_toggle() -> None:
    """Verify custom GPTLayerNorm handles layernorm bias configuration checks correctly."""
    C = 8
    ln_with_bias = GPTLayerNorm(ndim=C, bias=True)
    ln_without_bias = GPTLayerNorm(ndim=C, bias=False)

    assert ln_with_bias.bias is not None
    bias_attr = "bias"
    assert getattr(ln_without_bias, bias_attr) is None

    # Test forward pass with no bias
    x = torch.randn(2, 3, C)
    out = ln_without_bias(x)
    assert out.shape == (2, 3, C)


def test_layernorm_gradients() -> None:
    """Verify gradients flow correctly to weights and biases in custom GPTLayerNorm."""
    B, T, C = 2, 4, 8
    x = torch.randn(B, T, C)
    ln = GPTLayerNorm(ndim=C, bias=True)

    out = ln(x)
    loss = out.sum()
    loss.backward()

    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
    assert ln.weight.grad.shape == ln.weight.shape
    assert ln.bias.grad.shape == ln.bias.shape
