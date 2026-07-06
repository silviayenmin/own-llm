# Model Directory

This directory contains the custom-built Transformer architecture components. Every component is implemented from scratch using basic PyTorch tensor operations, with no usage of high-level modules like `nn.Transformer` or `nn.MultiheadAttention`.

## Modules

- `config.py`: Contains configurations matching the architecture settings.
- `embedding.py`: Custom token and positional embedding layer implementations.
- `attention.py`: Custom Multi-Head Self-Attention using raw matrix multiplication and masking.
- `feedforward.py`: Standard two-layer MLP for the transformer blocks.
- `layernorm.py`: Custom Layer Normalization implementation.
- `transformer_block.py`: Assembly of Attention, FeedForward, and LayerNorm with residual connections.
- `gpt.py`: The final GPT model combining embedding layers and stacked transformer blocks.
