from pathlib import Path

from tokenizer.tokenizer import BPETokenizer, CharTokenizer


def test_char_tokenizer_basic() -> None:
    """Verify that CharTokenizer correctly maps characters and handles round-trip decoding."""
    chars = ["a", "b", "c", "d", "e", " "]
    special_tokens = ["<|endoftext|>"]

    tokenizer = CharTokenizer(chars=chars, special_tokens=special_tokens)

    assert tokenizer.vocab_size == 7  # 6 standard chars + 1 special token

    # Test encoding
    text = "abc cba"
    ids = tokenizer.encode(text)
    assert len(ids) == len(text)

    # Test roundtrip
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_char_tokenizer_special_tokens() -> None:
    """Verify CharTokenizer handles special tokens and respects allowed_special flag."""
    chars = ["h", "e", "l", "o", " "]
    special_tokens = ["<|endoftext|>"]

    tokenizer = CharTokenizer(chars=chars, special_tokens=special_tokens)

    text = "hello <|endoftext|> hello"

    # With allowed_special = True (default)
    ids = tokenizer.encode(text, allowed_special=True)
    # "hello " -> 6 tokens, "<|endoftext|>" -> 1 token, " hello" -> 6 tokens. Total 13.
    assert len(ids) == 13
    assert tokenizer.special_stoi["<|endoftext|>"] in ids

    decoded = tokenizer.decode(ids)
    assert decoded == text

    # With allowed_special = False
    ids_no_special = tokenizer.encode(text, allowed_special=False)
    # The special token substring "<|endoftext|>" will have characters like '<', '|', 'e' etc.
    # which are not in standard chars, so they should be filtered out / ignored.
    # "hello " (6) + "e", "o", "e" (3) + " hello" (6) = 15 characters.
    assert len(ids_no_special) == 15
    decoded_no_special = tokenizer.decode(ids_no_special)
    assert decoded_no_special == "hello eoe hello"


def test_char_tokenizer_serialization(tmp_path: Path) -> None:
    """Verify CharTokenizer can be saved and loaded back correctly."""
    chars = ["x", "y", "z"]
    special_tokens = ["<|endoftext|>"]

    tokenizer = CharTokenizer(chars=chars, special_tokens=special_tokens)
    tokenizer.save(tmp_path)

    loaded = CharTokenizer.load(tmp_path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.chars == tokenizer.chars
    assert loaded.special_tokens == tokenizer.special_tokens

    text = "xyz"
    assert loaded.encode(text) == tokenizer.encode(text)


def test_bpe_tokenizer_training_and_roundtrip() -> None:
    """Verify training a BPETokenizer and that encoded strings roundtrip perfectly."""
    text = "hello world! hello universe! this is a custom BPE test."
    tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])

    # Train BPE
    target_vocab_size = 300
    tokenizer.train(text, vocab_size=target_vocab_size)

    assert tokenizer.vocab_size == target_vocab_size

    # Test roundtrip on trained text
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text

    # Test roundtrip on unseen text with unicode characters
    unseen = "hello python! 🌟 unicode emoji test."
    ids_unseen = tokenizer.encode(unseen)
    decoded_unseen = tokenizer.decode(ids_unseen)
    assert decoded_unseen == unseen


def test_bpe_tokenizer_special_tokens() -> None:
    """Verify special tokens behavior in BPETokenizer."""
    text = "hello <|endoftext|> world"
    tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])

    # Let's train to add a few merges
    tokenizer.train("hello world. hello world. hello world.", vocab_size=265)

    # allowed_special = True
    ids_special = tokenizer.encode(text, allowed_special=True)
    assert tokenizer.special_stoi["<|endoftext|>"] in ids_special
    assert tokenizer.decode(ids_special) == text

    # allowed_special = False
    # "<|endoftext|>" should be split into individual byte tokens instead of being matched
    ids_no_special = tokenizer.encode(text, allowed_special=False)
    assert tokenizer.special_stoi["<|endoftext|>"] not in ids_no_special
    assert tokenizer.decode(ids_no_special) == text


def test_bpe_tokenizer_serialization(tmp_path: Path) -> None:
    """Verify BPETokenizer saves and loads its merges and configuration properly."""
    corpus = "ab cd ab cd ab cd ab cd"
    tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])
    tokenizer.train(corpus, vocab_size=260)

    tokenizer.save(tmp_path)

    loaded = BPETokenizer.load(tmp_path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.special_tokens == tokenizer.special_tokens

    test_str = "ab cd"
    assert loaded.encode(test_str) == tokenizer.encode(test_str)
    assert loaded.decode(loaded.encode(test_str)) == test_str
