import os
import shutil
import tempfile
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from api.main import app
from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.tokenizer import CharTokenizer

def test_api_root_html() -> None:
    """Verify that GET / returns the HTML interface successfully."""
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Mini GPT Playground" in res.text


def test_api_health_degraded() -> None:
    """Verify that if no checkpoints exist, API starts in degraded health mode and rejects generate requests."""
    # Point checkpoint and tokenizer paths to non-existent locations
    env_patches = {
        "MODEL_CHECKPOINT_PATH": "non_existent_checkpoint_file.pt",
        "TOKENIZER_DIR": "non_existent_tokenizer_dir",
    }
    with patch.dict(os.environ, env_patches):
        # Use lifespan-supportive TestClient context
        with TestClient(app) as client:
            # 1) Get health checks
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False
            assert data["tokenizer_loaded"] is False

            # 2) Generate request should fail with 503 Service Unavailable
            res_gen = client.post(
                "/generate",
                json={
                    "prompt": "hello",
                    "max_new_tokens": 5,
                },
            )
            assert res_gen.status_code == 503
            assert "Model is not loaded" in res_gen.json()["detail"]


def test_api_generate_healthy() -> None:
    """Verify that with valid checkpoints, API starts in healthy mode and generates tokens successfully."""
    temp_dir = tempfile.mkdtemp()
    try:
        # 1) Initialize and save a mock CharTokenizer
        chars = ["a", "b", "c", "d", " "]
        tokenizer = CharTokenizer(chars=chars, special_tokens=["<|endoftext|>"])
        tokenizer_dir = os.path.join(temp_dir, "tokenizer")
        tokenizer.save(tokenizer_dir)

        # 2) Initialize and save a mock model checkpoint
        config = GPTConfig()
        config.model.vocab_size = tokenizer.vocab_size
        config.model.n_embd = 16
        config.model.n_head = 2
        config.model.n_layer = 1
        config.model.block_size = 8
        model = GPT(config)

        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "step": 0,
            "val_loss": 0.0,
            "config": config.model_dump(),
        }
        torch.save(checkpoint, checkpoint_path)

        # 3) Patch the server configuration environment variables to point to temp directories
        env_patches = {
            "MODEL_CHECKPOINT_PATH": checkpoint_path,
            "TOKENIZER_DIR": tokenizer_dir,
        }
        with patch.dict(os.environ, env_patches):
            with TestClient(app) as client:
                # Health checks verify status is healthy
                res = client.get("/health")
                assert res.status_code == 200
                data = res.json()
                assert data["status"] == "healthy"
                assert data["model_loaded"] is True
                assert data["tokenizer_loaded"] is True

                # POST /generate endpoint works and completes prompt sequence
                res_gen = client.post(
                    "/generate",
                    json={
                        "prompt": "abc",
                        "max_new_tokens": 5,
                        "temperature": 0.8,
                        "top_k": 2,
                    },
                )
                assert res_gen.status_code == 200
                res_data = res_gen.json()
                assert res_data["prompt"] == "abc"
                assert "generated_text" in res_data
                assert res_data["num_tokens"] == 5

    finally:
        shutil.rmtree(temp_dir)
