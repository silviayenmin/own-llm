from typing import cast

import pytest
import torch

from model.config import GPTConfig
from model.gpt import GPT
from model.transformer_block import GPTTransformerBlock


def test_gpt_forward_no_targets() -> None:
    """Verify GPT model forward pass output shape without targets."""
    config = GPTConfig()
    config.model.vocab_size = 100
    config.model.n_layer = 2
    config.model.n_head = 2
    config.model.n_embd = 16
    config.model.block_size = 8

    model = GPT(config)

    B, T = 2, 4
    idx = torch.randint(0, config.model.vocab_size, (B, T))
    logits, loss = model(idx)

    assert logits.shape == (B, T, config.model.vocab_size)
    assert loss is None


def test_gpt_forward_with_targets() -> None:
    """Verify GPT model forward pass returns correct loss and shape when targets are provided."""
    config = GPTConfig()
    config.model.vocab_size = 50
    config.model.n_layer = 1
    config.model.n_head = 2
    config.model.n_embd = 8
    config.model.block_size = 8

    model = GPT(config)

    B, T = 2, 4
    idx = torch.randint(0, config.model.vocab_size, (B, T))
    targets = torch.randint(0, config.model.vocab_size, (B, T))

    logits, loss = model(idx, targets)

    assert logits.shape == (B, T, config.model.vocab_size)
    assert loss is not None
    assert loss.shape == ()  # Scalar tensor

    # Manual loss calculation verify
    expected_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1
    )
    assert torch.allclose(loss, expected_loss)


def test_gpt_weight_tying() -> None:
    """Verify that output language modeling head shares memory with the input token embeddings."""
    config = GPTConfig()
    config.model.vocab_size = 100
    config.model.n_embd = 16
    config.model.n_head = 2
    config.model.n_layer = 1
    model = GPT(config)

    # Modify weight of token embeddings directly
    with torch.no_grad():
        model.transformer.wte.embedding.weight[0, 0] = 999.0

    # Output head weights should be modified as well due to weight tying/sharing
    assert model.lm_head.weight[0, 0] == 999.0
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.embedding.weight.data_ptr()


def test_gpt_crop_block_size() -> None:
    """Verify that crop_block_size correctly shrinks context limits and causal masks."""
    config = GPTConfig()
    config.model.vocab_size = 50
    config.model.n_embd = 16
    config.model.n_head = 2
    config.model.n_layer = 2
    config.model.block_size = 16
    model = GPT(config)

    # Initial checks
    assert model.config.model.block_size == 16
    assert model.transformer.wpe.embedding.weight.shape[0] == 16

    # Crop to size 8
    model.crop_block_size(8)

    assert model.config.model.block_size == 8
    assert model.transformer.wpe.embedding.weight.shape[0] == 8

    # Verify that attention mask buffers are also cropped
    for block in model.transformer.h:
        block_module = cast(GPTTransformerBlock, block)
        bias_buffer = cast(torch.Tensor, block_module.attn.bias)
        assert bias_buffer is not None
        assert bias_buffer.shape[-1] == 8
        assert bias_buffer.shape[-2] == 8

    # Sequence of length 10 should now fail forward validation check
    idx = torch.randint(0, 50, (1, 10))
    with pytest.raises(ValueError):
        model(idx)


def test_gpt_gradients() -> None:
    """Verify that gradients correctly backpropagate and accumulate throughout all GPT parameters."""
    config = GPTConfig()
    config.model.vocab_size = 50
    config.model.n_embd = 16
    config.model.n_layer = 2
    config.model.n_head = 2
    model = GPT(config)

    idx = torch.randint(0, 50, (2, 4))
    targets = torch.randint(0, 50, (2, 4))
    logits, loss = model(idx, targets)

    assert loss is not None
    loss.backward()

    # Check gradients in embeddings, blocks and projection layers
    assert model.transformer.wte.embedding.weight.grad is not None
    assert model.transformer.wpe.embedding.weight.grad is not None
    first_block = cast(GPTTransformerBlock, model.transformer.h[0])
    assert first_block.attn.c_attn.weight.grad is not None
    assert first_block.mlp.c_fc.weight.grad is not None
    assert model.transformer.ln_f.weight.grad is not None

    # Check output projections
    assert model.lm_head.weight.grad is not None
