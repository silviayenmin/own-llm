"""Custom tokenizers implemented from scratch for Mini GPT.

This module provides two tokenizer implementations:
1. CharTokenizer: Character-level tokenization (simple, character-to-index mapping).
2. BPETokenizer: Byte-Pair Encoding tokenization (byte-level BPE from scratch).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):
    """Abstract base class defining the shared interface for all tokenizers."""

    @abstractmethod
    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode a string into a list of token IDs.

        Args:
            text: The raw input string.
            allowed_special: Whether to process special tokens or treat them as regular text.

        Returns:
            A list of integer token IDs.
        """
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        pass

    @abstractmethod
    def save(self, directory: str | Path) -> None:
        """Save the tokenizer vocabulary and settings to a directory.

        Args:
            directory: Target directory path.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> BaseTokenizer:
        """Load a tokenizer instance from a directory.

        Args:
            directory: Source directory path.

        Returns:
            An instance of the loaded tokenizer.
        """
        pass


class CharTokenizer(BaseTokenizer):
    """Simple character-level tokenizer mapping characters to integer IDs."""

    def __init__(
        self,
        chars: list[str] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Initialize the character tokenizer.

        Args:
            chars: A list of unique characters in the vocabulary.
            special_tokens: A list of special tokens (e.g., ["<|endoftext|>"]).
        """
        self.special_tokens = special_tokens or []
        self.chars = chars or []

        # Build mapping dictionaries
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self._rebuild_mappings()

    def _rebuild_mappings(self) -> None:
        # Assign IDs to unique characters
        self.stoi = {char: i for i, char in enumerate(self.chars)}

        # Assign IDs to special tokens (starting after the standard characters)
        start_idx = len(self.chars)
        self.special_stoi = {
            token: start_idx + i for i, token in enumerate(self.special_tokens)
        }
        self.stoi.update(self.special_stoi)

        # Invert mapping
        self.itos = {i: s for s, i in self.stoi.items()}

    def _encode_chunk(self, text: str) -> list[int]:
        ids = []
        for char in text:
            if char in self.stoi:
                ids.append(self.stoi[char])
            else:
                # Silently ignore or map to unknown character if not in vocab
                pass
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if not text:
            return []

        if not self.special_tokens or not allowed_special:
            return self._encode_chunk(text)

        # Split by special tokens to prevent them from being tokenized as individual characters
        escaped_specials = [re.escape(t) for t in self.special_tokens]
        pattern = re.compile(f"({'|'.join(escaped_specials)})")
        parts = pattern.split(text)

        ids = []
        for part in parts:
            if part in self.special_stoi:
                ids.append(self.special_stoi[part])
            else:
                ids.extend(self._encode_chunk(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(idx, "") for idx in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "type": "char",
            "chars": self.chars,
            "special_tokens": self.special_tokens,
        }
        with open(path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str | Path) -> CharTokenizer:
        path = Path(directory)
        with open(path / "tokenizer.json", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("type") != "char":
            raise ValueError(f"Expected char tokenizer config, got type: {config.get('type')}")
        return cls(
            chars=config.get("chars"),
            special_tokens=config.get("special_tokens"),
        )


def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Calculate frequencies of adjacent pairs of token IDs."""
    stats: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:], strict=False):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace occurrences of a pair of token IDs with a new token ID."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer(BaseTokenizer):
    """Custom Byte-Pair Encoding (BPE) tokenizer implemented from scratch."""

    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Initialize the BPE tokenizer.

        Args:
            merges: Dictionary mapping pairs of token IDs to their merged token ID.
            special_tokens: List of special tokens to support.
        """
        self.merges = merges or {}
        self.special_tokens = special_tokens or []
        self._build_vocab()

    def _build_vocab(self) -> None:
        # Build initial byte-level vocabulary (0-255 map to their byte values)
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Map special tokens starting from 256
        self.special_stoi: dict[str, int] = {}
        self.special_itos: dict[int, str] = {}
        for i, token in enumerate(self.special_tokens):
            idx = 256 + i
            self.special_stoi[token] = idx
            self.special_itos[idx] = token
            self.vocab[idx] = token.encode("utf-8")

        # Build vocabulary from merges
        # Note: merges are applied sequentially. We assume merges values (new IDs) start
        # after 256 + len(special_tokens).
        sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])
        for (p0, p1), idx in sorted_merges:
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the BPE tokenizer on a corpus.

        Args:
            text: Raw training text.
            vocab_size: Target vocabulary size.
            verbose: If True, prints progress details.
        """
        if vocab_size <= 256 + len(self.special_tokens):
            raise ValueError(
                f"vocab_size {vocab_size} must be greater than base vocab + special tokens "
                f"({256 + len(self.special_tokens)})"
            )

        num_merges = vocab_size - 256 - len(self.special_tokens)

        # Convert text to standard UTF-8 byte tokens
        tokens = list(text.encode("utf-8"))

        self.merges = {}
        current_vocab = {i: bytes([i]) for i in range(256)}

        # Special tokens vocabulary
        for i, token in enumerate(self.special_tokens):
            idx = 256 + i
            current_vocab[idx] = token.encode("utf-8")

        start_merge_idx = 256 + len(self.special_tokens)

        for i in range(num_merges):
            stats = _get_stats(tokens)
            if not stats:
                if verbose:
                    print(f"Stopping early at step {i} due to no adjacent pairs.")
                break

            # Find the most frequent pair
            best_pair = max(stats, key=stats.get) # type: ignore
            new_id = start_merge_idx + i

            if verbose:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id} (count: {stats[best_pair]})")

            tokens = _merge(tokens, best_pair, new_id)
            self.merges[best_pair] = new_id
            current_vocab[new_id] = current_vocab[best_pair[0]] + current_vocab[best_pair[1]]

        # Re-initialize to populate self.vocab, self.stoi, etc.
        self._build_vocab()

    def _encode_chunk(self, text: str) -> list[int]:
        # If text is large, chunk it to avoid O(N^2) list operation slowdowns in pure Python
        chunk_size = 1000
        if len(text) > chunk_size:
            ids = []
            for i in range(0, len(text), chunk_size):
                ids.extend(self._encode_chunk(text[i : i + chunk_size]))
            return ids

        # Convert chunk to UTF-8 bytes representation
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            # Find the merge rule that has the lowest merge ID (highest priority)
            stats = _get_stats(ids)
            pair_to_merge = None
            min_merge_idx = float("inf")

            for pair in stats:
                if pair in self.merges:
                    merge_idx = self.merges[pair]
                    if merge_idx < min_merge_idx:
                        min_merge_idx = merge_idx
                        pair_to_merge = pair

            if pair_to_merge is None:
                break # No more merge rules apply

            ids = _merge(ids, pair_to_merge, self.merges[pair_to_merge])

        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if not text:
            return []

        if not self.special_tokens or not allowed_special:
            return self._encode_chunk(text)

        # Split by special tokens
        escaped_specials = [re.escape(t) for t in self.special_tokens]
        pattern = re.compile(f"({'|'.join(escaped_specials)})")
        parts = pattern.split(text)

        ids = []
        for part in parts:
            if part in self.special_stoi:
                ids.append(self.special_stoi[part])
            else:
                ids.extend(self._encode_chunk(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw_bytes = bytearray()
        for idx in ids:
            if idx in self.vocab:
                raw_bytes.extend(self.vocab[idx])
            else:
                # Ignore unknown token IDs
                pass
        return raw_bytes.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Serialize merges dict by converting tuples of ints to space-separated string key
        # e.g., (12, 34) -> "12 34"
        merges_serialized = {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}

        config = {
            "type": "bpe",
            "special_tokens": self.special_tokens,
            "merges": merges_serialized,
        }
        with open(path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str | Path) -> BPETokenizer:
        path = Path(directory)
        with open(path / "tokenizer.json", encoding="utf-8") as f:
            config = json.load(f)

        if config.get("type") != "bpe":
            raise ValueError(f"Expected bpe tokenizer config, got type: {config.get('type')}")

        # Parse merges back to dict[tuple[int, int], int]
        merges = {}
        for k, v in config.get("merges", {}).items():
            p0, p1 = map(int, k.split())
            merges[(p0, p1)] = v

        return cls(
            merges=merges,
            special_tokens=config.get("special_tokens"),
        )
