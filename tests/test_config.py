from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from model.config import DataConfig, GPTConfig, ModelConfig, TrainingConfig


def test_default_config_loading() -> None:
    """Verify that default model, training, and data configurations can be loaded without issues."""
    config = GPTConfig()
    assert config.model.vocab_size == 50304
    assert config.model.block_size == 512
    assert config.model.n_layer == 6
    assert config.model.n_head == 6
    assert config.model.n_embd == 384
    assert config.model.dropout == 0.1
    assert not config.model.bias

    assert config.training.batch_size == 64
    assert config.training.device == "cuda"
    assert config.data.train_split == 0.9


def test_from_yaml_valid(tmp_path: Path) -> None:
    """Verify loading config values correctly overrides defaults using a temporary YAML file."""
    yaml_content = {
        "model": {
            "vocab_size": 10000,
            "block_size": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "dropout": 0.0,
            "bias": True
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "device": "cpu"
        },
        "data": {
            "train_split": 0.8
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(yaml_content, f)

    config = GPTConfig.from_yaml(config_file)
    assert config.model.vocab_size == 10000
    assert config.model.block_size == 256
    assert config.model.n_layer == 4
    assert config.model.n_head == 4
    assert config.model.n_embd == 128
    assert config.model.dropout == 0.0
    assert config.model.bias

    assert config.training.batch_size == 32
    assert config.training.learning_rate == 1e-3
    assert config.training.device == "cpu"

    assert config.data.train_split == 0.8


def test_from_yaml_file_not_found() -> None:
    """Verify from_yaml raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        GPTConfig.from_yaml("non_existent_config_file_path.yaml")


def test_validation_out_of_bounds() -> None:
    """Verify Pydantic validation raises errors when parameters violate numerical bounds."""
    # Test invalid learning rate (<= 0)
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=0.0)

    # Test invalid dropout (> 1.0)
    with pytest.raises(ValidationError):
        ModelConfig(dropout=1.1)

    # Test invalid layers (<= 0)
    with pytest.raises(ValidationError):
        ModelConfig(n_layer=0)

    # Test invalid data split (>= 1.0)
    with pytest.raises(ValidationError):
        DataConfig(train_split=1.0)


def test_to_dict() -> None:
    """Verify that configuration correctly converts to a standard dictionary representation."""
    config = GPTConfig()
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "model" in config_dict
    assert "training" in config_dict
    assert "data" in config_dict
    assert config_dict["model"]["vocab_size"] == 50304
