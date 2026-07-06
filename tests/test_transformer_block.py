import torch
import torch.nn as nn

from model.config import ModelConfig
from model.transformer_block import GPTTransformerBlock


def test_transformer_block_shape() -> None:
    """Verify custom GPTTransformerBlock preserves dimension shapes correctly."""
    config = ModelConfig(n_embd=32, n_head=4, block_size=16, dropout=0.1)
    block = GPTTransformerBlock(config)

    B, T, C = 2, 8, 32
    x = torch.randn(B, T, C)
    out = block(x)

    assert out.shape == (B, T, C)


def test_transformer_block_residual_identity() -> None:
    """Verify that if sub-layers are initialized to zero, the block acts as an identity mapping."""
    config = ModelConfig(n_embd=16, n_head=2, block_size=8, dropout=0.0, bias=True)
    block = GPTTransformerBlock(config)

    # Set projection weights of self-attention and mlp to zero
    with torch.no_grad():
        block.attn.c_proj.weight.zero_()
        if block.attn.c_proj.bias is not None:
            block.attn.c_proj.bias.zero_()

        block.mlp.c_proj.weight.zero_()
        if block.mlp.c_proj.bias is not None:
            block.mlp.c_proj.bias.zero_()

    x = torch.randn(2, 4, 16)
    out = block(x)

    # Due to residual connection, out should match x exactly
    assert torch.allclose(out, x, atol=1e-6)


def test_transformer_block_gradients() -> None:
    """Verify gradients propagate correctly to attention and mlp parameters in the block."""
    config = ModelConfig(n_embd=16, n_head=2, bias=True)
    block = GPTTransformerBlock(config)

    x = torch.randn(2, 4, 16)
    out = block(x)
    loss = out.sum()
    loss.backward()

    # Verify gradients are accumulated on components parameters
    assert block.attn.c_attn.weight.grad is not None
    assert block.mlp.c_fc.weight.grad is not None
    assert block.ln_1.weight.grad is not None
    assert block.ln_2.weight.grad is not None


def test_transformer_block_stacking() -> None:
    """Verify that multiple stacked GPTTransformerBlocks run correctly."""
    config = ModelConfig(n_embd=16, n_head=2, bias=True)

    # Stack 3 transformer blocks
    num_layers = 3
    blocks = nn.ModuleList([GPTTransformerBlock(config) for _ in range(num_layers)])

    x = torch.randn(2, 4, 16)

    # Run sequential pass through the stack
    for block in blocks:
        x = block(x)

    assert x.shape == (2, 4, 16)
