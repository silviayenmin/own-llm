"""Full decoder-only GPT model assembly.

Combines token embeddings, positional embeddings, stacked transformer blocks,
and a language modeling output projection head with weight sharing.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig, ModelConfig
from model.embedding import GPTPositionalEmbedding, GPTTokenEmbedding
from model.layernorm import GPTLayerNorm
from model.transformer_block import GPTTransformerBlock


class GPTTransformer(nn.Module):
    """Container module for GPT transformer layers to ensure clean static typing."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = GPTTokenEmbedding(config)
        self.wpe = GPTPositionalEmbedding(config)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([GPTTransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = GPTLayerNorm(config.n_embd, bias=config.bias)


class GPT(nn.Module):
    """Decoder-only GPT Transformer architecture implemented from scratch."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the GPT model.

        Args:
            config: Unified GPTConfig configuration parameters.
        """
        super().__init__()
        self.config = config

        # Assembly container for transformer layers
        self.transformer = GPTTransformer(config.model)

        # Language modeling projection head (maps n_embd back to vocab_size logits)
        self.lm_head = nn.Linear(config.model.n_embd, config.model.vocab_size, bias=False)

        # Weight Sharing: wte and lm_head share weights (GPT-2 standard)
        # Note: self.transformer.wte is a GPTTokenEmbedding, containing a self.embedding nn.Embedding
        self.lm_head.weight = self.transformer.wte.embedding.weight

        # Recursively initialize all weights
        self.apply(self._init_weights)

        # Scale weights of residual output projections (c_proj in attention and MLP)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.model.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize parameters using normal distributions matching GPT-2 settings.

        Args:
            module: The PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the GPT model.

        Args:
            idx: Integer tensor of shape (B, T) containing token IDs.
            targets: Optional ground-truth token IDs tensor of shape (B, T) for loss calculation.

        Returns:
            Tuple of:
              - logits: Output token logit tensor of shape (B, T, vocab_size).
              - loss: Cross entropy loss scalar tensor, or None if targets are not provided.
        """
        _, t = idx.size()
        if t > self.config.model.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.model.block_size}"
            )

        # Compute embeddings
        tok_emb = self.transformer.wte(idx)  # Shape: (B, T, n_embd)
        pos_emb = self.transformer.wpe(idx)  # Shape: (1, T, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward pass through stacked Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Apply final Layer Normalization
        x = self.transformer.ln_f(x)

        # Map to vocabulary logits
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten B and T dimensions to compute CrossEntropyLoss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Crop the maximum context window of the model down to a smaller size.

        Args:
            block_size: Target cropped context window size.
        """
        if block_size > self.config.model.block_size:
            raise ValueError(
                f"Cannot crop to block size {block_size} which is greater "
                f"than current block size {self.config.model.block_size}."
            )

        self.config.model.block_size = block_size

        # Crop position embedding look-up table weight parameter
        wpe_weight = self.transformer.wpe.embedding.weight
        self.transformer.wpe.embedding.weight = nn.Parameter(wpe_weight[:block_size])

        # Crop causal mask buffers in all self-attention modules
        for block in self.transformer.h:
            # Type cast to GPTTransformerBlock layer to access attn
            block_module = cast(GPTTransformerBlock, block)
            attn = block_module.attn
            if hasattr(attn, "bias"):
                bias_buffer = cast(torch.Tensor, attn.bias)
                # Crop mask buffer along sequence dimensions
                cropped_bias = bias_buffer[:, :, :block_size, :block_size]
                attn.register_buffer("bias", cropped_bias)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens given a conditioning prefix prompt.

        Args:
            idx: Input token IDs tensor of shape (B, T).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Softmax scaling factor. Higher means more random, lower means more deterministic.
            top_k: Optional top-k filter keeping only the top k tokens with the highest probabilities.

        Returns:
            The input tensor appended with max_new_tokens newly generated tokens.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it to block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.model.block_size
                else idx[:, -self.config.model.block_size :]
            )

            # Forward pass to get logits for the final step
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on the last token step: shape (B, vocab_size)

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / max(1e-5, temperature)

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
