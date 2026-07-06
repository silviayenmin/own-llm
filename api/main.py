"""FastAPI server wrapping the trained GPT model for real-time text generation.

Provides a POST /generate endpoint for text inference, and a GET /health
endpoint for server status checking.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from generate import load_tokenizer
from model.config import GPTConfig
from model.gpt import GPT

# Global state container for tokenizer and loaded model parameters
global_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": "cpu",
    "model_loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup model loading and shutdown cleanup."""
    checkpoint_path = os.getenv(
        "MODEL_CHECKPOINT_PATH", "experiments/checkpoints/best_model.pt"
    )
    tokenizer_dir = os.getenv("TOKENIZER_DIR", "data/tokenizer")

    # 1) Try loading custom BPE or Character tokenizer
    try:
        if os.path.exists(tokenizer_dir):
            global_state["tokenizer"] = load_tokenizer(tokenizer_dir)
            print(f"Successfully loaded tokenizer from {tokenizer_dir}")
        else:
            print(
                f"Warning: Tokenizer directory {tokenizer_dir} does not exist. "
                "Tokenizer not loaded."
            )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

    # 2) Try loading GPT model checkpoint
    try:
        if os.path.exists(checkpoint_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            global_state["device"] = device
            print(
                f"Loading model checkpoint from {checkpoint_path} onto {device}..."
            )

            checkpoint = torch.load(checkpoint_path, map_location=device)
            config_dict = checkpoint["config"]
            config = GPTConfig(**config_dict)
            config.training.device = device

            model = GPT(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            global_state["model"] = model
            global_state["model_loaded"] = True
            print("Model loaded successfully and ready for inference!")
        else:
            print(
                f"Warning: Model checkpoint file not found at {checkpoint_path}. "
                "Inference service will degrade to 503 errors until a model is placed."
            )
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")

    yield

    # Cleanup resources on shutdown
    global_state["model"] = None
    global_state["tokenizer"] = None
    global_state["model_loaded"] = False
    print("Application shutdown clean-up completed.")


# Initialize FastAPI app with lifespans
app = FastAPI(
    title="Mini GPT Inference API",
    description="Real-time text generation endpoints for custom trained decoder-only GPT model.",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    """Pydantic schema mapping text generation request fields."""

    prompt: str = Field(
        default="", description="Input text prompt string to seed token generation."
    )
    max_new_tokens: int = Field(
        default=100,
        description="Maximum number of new tokens to append.",
        gt=0,
        le=1000,
    )
    temperature: float = Field(
        default=1.0,
        description="Softmax temperature scaling factor controlling output randomness.",
        gt=0.0,
    )
    top_k: int | None = Field(
        default=None,
        description="Optional Top-K filtering count limiting token options.",
        gt=0,
    )


class GenerateResponse(BaseModel):
    """Pydantic schema mapping generation response fields."""

    prompt: str
    generated_text: str
    num_tokens: int


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini GPT Playground</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f17;
            --panel-bg: rgba(22, 28, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #6366f1;
            --accent-secondary: #a855f7;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 2rem 2rem 1rem 2rem;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(90deg, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }

        .header-title p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .status-badge {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: var(--success-color);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .status-badge.degraded {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: var(--danger-color);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: currentColor;
            box-shadow: 0 0 12px currentColor;
        }

        main {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 1rem 2rem 3rem 2rem;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
            margin-bottom: 0.5rem;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-primary);
            display: flex;
            justify-content: space-between;
        }

        .value-display {
            color: var(--accent-primary);
            font-weight: 600;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 9999px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-gradient);
            cursor: pointer;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
            transition: transform 0.1s ease;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        .input-text, select {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            padding: 0.75rem;
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .input-text:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        textarea.input-text {
            resize: vertical;
            min-height: 120px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
        }

        .chips-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 9999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.8rem;
            cursor: pointer;
            color: var(--text-secondary);
            transition: all 0.2s ease;
        }

        .chip:hover {
            background: rgba(99, 102, 241, 0.15);
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        button.btn-primary {
            background: var(--accent-gradient);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            padding: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }

        button.btn-primary:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }

        button.btn-primary:active {
            transform: translateY(1px);
        }

        button.btn-primary:disabled {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .system-meta {
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem 0.5rem;
        }

        .meta-item {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            background: rgba(0, 0, 0, 0.15);
            padding: 0.5rem;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        .meta-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6b7280;
        }

        .meta-val {
            font-weight: 600;
            color: var(--text-primary);
        }

        .playground {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .output-container {
            flex: 1;
            min-height: 250px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            line-height: 1.6;
            overflow-y: auto;
            white-space: pre-wrap;
            position: relative;
            transition: all 0.3s ease;
        }

        .output-container.loading {
            border-color: rgba(99, 102, 241, 0.3);
        }

        .output-container.loading::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: var(--accent-gradient);
            animation: pulse-border 1.5s infinite linear;
        }

        @keyframes pulse-border {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .placeholder-text {
            color: var(--text-secondary);
            font-style: italic;
        }

        .metric-banner {
            display: flex;
            gap: 1rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            align-items: center;
        }

        .metric-badge {
            background: rgba(99, 102, 241, 0.1);
            color: var(--accent-primary);
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }

        .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Guide Drawer/Panel styling */
        .tabs-header {
            display: flex;
            border-bottom: 1px solid var(--border-color);
        }
        .tab-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 0.75rem 1.25rem;
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }
        .tab-btn.active {
            color: var(--text-primary);
            border-bottom-color: var(--accent-primary);
        }
        .tab-content {
            display: none;
            padding-top: 1rem;
        }
        .tab-content.active {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .guide-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }
        .guide-box h4 {
            color: var(--accent-primary);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        .guide-box code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.4);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #ff79c6;
        }
        .guide-box pre {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.4);
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            overflow-x: auto;
            margin-top: 0.5rem;
            border: 1px solid rgba(255,255,255,0.03);
            color: #f8f8f2;
        }
        .guide-box ol, .guide-box ul {
            padding-left: 1.25rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>🌌 Mini GPT Playground</h1>
            <p>Interactive testing & generation suite for custom-trained GPT models</p>
        </div>
        <div id="service-status" class="status-badge degraded">
            <span class="status-dot"></span>
            <span id="status-text">Connecting...</span>
        </div>
    </header>

    <main>
        <!-- Sidebar controls -->
        <div class="card">
            <div class="section-title">Model Settings</div>

            <!-- Param sliders -->
            <div class="control-group">
                <label for="temperature">
                    <span>Temperature</span>
                    <span id="temp-val" class="value-display">1.0</span>
                </label>
                <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('temp-val').innerText = this.value">
                <span style="font-size: 0.75rem; color: var(--text-secondary);">Controls text randomness (lower is more focused/predictable).</span>
            </div>

            <div class="control-group">
                <label for="max-tokens">
                    <span>Max New Tokens</span>
                    <span id="tokens-val" class="value-display">100</span>
                </label>
                <input type="range" id="max-tokens" min="1" max="500" step="10" value="100" oninput="document.getElementById('tokens-val').innerText = this.value">
                <span style="font-size: 0.75rem; color: var(--text-secondary);">Maximum number of tokens the model will generate.</span>
            </div>

            <div class="control-group">
                <label for="top-k">
                    <span>Top-K Sampling (Optional)</span>
                    <span id="topk-val" class="value-display">None</span>
                </label>
                <input type="range" id="top-k" min="0" max="100" step="1" value="0" oninput="document.getElementById('topk-val').innerText = this.value == 0 ? 'None' : this.value">
                <span style="font-size: 0.75rem; color: var(--text-secondary);">Limits token pool to the K most probable options.</span>
            </div>

            <!-- Health check specs -->
            <div class="section-title">API Meta Details</div>
            <div class="system-meta">
                <div class="meta-item">
                    <span class="meta-label">Device</span>
                    <span id="meta-device" class="meta-val">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Model Checkpoint</span>
                    <span id="meta-model" class="meta-val">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Tokenizer</span>
                    <span id="meta-tokenizer" class="meta-val">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Vocab size</span>
                    <span id="meta-vocab" class="meta-val">Custom BPE</span>
                </div>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="playground">
            <div class="card" style="flex: 1;">
                <div class="tabs-header">
                    <button class="tab-btn active" onclick="switchTab('play')">Playground</button>
                    <button class="tab-btn" onclick="switchTab('guide')">Train & Test Guide</button>
                </div>

                <!-- Tab 1: Playground UI -->
                <div id="tab-play" class="tab-content active">
                    <div class="control-group">
                        <label for="prompt">Prompt Seed Text</label>
                        <textarea id="prompt" class="input-text" placeholder="First Citizen:&#10;Before we proceed any further, hear me speak."></textarea>
                        
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">
                            Quick Seed Chips:
                            <div class="chips-container">
                                <span class="chip" onclick="setPrompt('First Citizen:')">First Citizen:</span>
                                <span class="chip" onclick="setPrompt('Speak, speak.')">Speak, speak.</span>
                                <span class="chip" onclick="setPrompt('Second Citizen:\\nOne word, good citizens.')">Second Citizen: ...</span>
                                <span class="chip" onclick="setPrompt('&lt;|endoftext|&gt;')">&lt;|endoftext|&gt; (BOS)</span>
                            </div>
                        </div>
                    </div>

                    <button id="generate-btn" class="btn-primary" onclick="generateText()">
                        <span class="spinner" id="btn-spinner"></span>
                        <span id="btn-text">Generate Completion</span>
                    </button>

                    <div class="control-group" style="flex: 1; display: flex; flex-direction: column; gap: 0.5rem;">
                        <label>Generated Text Output</label>
                        <div id="output" class="output-container">
                            <span class="placeholder-text">Generated text will appear here...</span>
                        </div>
                    </div>

                    <div class="metric-banner" id="metric-banner" style="display: none;">
                        <span>Generated <span id="metric-tokens" class="metric-badge">0</span> tokens</span>
                        <span>|</span>
                        <span>Latency: <span id="metric-time" class="metric-badge">0 ms</span></span>
                    </div>
                </div>

                <!-- Tab 2: Instruction Guide -->
                <div id="tab-guide" class="tab-content">
                    <div class="guide-box">
                        <h4>1. Prepare data & Tokenizer</h4>
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Train a BPE or Char tokenizer, then tokenize raw text into binary formats:</p>
                        <pre># Train tokenizer (character or BPE)
python tokenizer/train_tokenizer.py --type bpe --vocab_size 320

# Preprocess raw corpus (produces train.bin/val.bin)
python -c "from model.config import GPTConfig; from tokenizer.tokenizer import BPETokenizer; from training.dataset import preprocess_data; config = GPTConfig(); tokenizer = BPETokenizer.load('data/tokenizer'); preprocess_data(config, tokenizer)"</pre>
                    </div>

                    <div class="guide-box">
                        <h4>2. Execute Model Training</h4>
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Launch training using the configuration specifications. For quick testing, run the tiny config on CPU:</p>
                        <pre># Train a tiny model for validation (Quick CPU run)
python training/train.py --config configs/config_tiny.yaml

# Train standard model parameters (custom configs)
python training/train.py --config configs/config.yaml</pre>
                    </div>

                    <div class="guide-box">
                        <h4>3. Test locally using CLI</h4>
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Generate text outputs straight from the command line:</p>
                        <pre>python generate.py --checkpoint experiments/checkpoints/best_model.pt --prompt "First Citizen:"</pre>
                    </div>

                    <div class="guide-box">
                        <h4>4. Run pytest test suite</h4>
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Validate code correctness against unit test assertions:</p>
                        <pre>python -m pytest</pre>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer style="text-align: center; padding: 1.5rem; color: var(--text-secondary); font-size: 0.8rem; border-top: 1px solid var(--border-color); margin-top: auto;">
        Mini GPT &bull; Implemented from scratch in PyTorch &bull; FastAPI Deployment
    </footer>

    <script>
        // Check API health on load
        async function checkHealth() {
            const statusBadge = document.getElementById('service-status');
            const statusText = document.getElementById('status-text');
            const generateBtn = document.getElementById('generate-btn');
            
            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                if (data.status === 'healthy' || data.model_loaded) {
                    statusBadge.className = 'status-badge';
                    statusText.innerText = 'Online / Ready';
                    generateBtn.removeAttribute('disabled');
                    
                    document.getElementById('meta-device').innerText = data.device || 'CPU';
                    document.getElementById('meta-model').innerText = 'Loaded (best_model.pt)';
                    document.getElementById('meta-tokenizer').innerText = data.tokenizer_loaded ? 'Loaded' : 'Not Loaded';
                } else {
                    statusBadge.className = 'status-badge degraded';
                    statusText.innerText = 'Degraded (No model)';
                    
                    document.getElementById('meta-device').innerText = data.device || 'CPU';
                    document.getElementById('meta-model').innerText = 'Missing / Outdated';
                    document.getElementById('meta-tokenizer').innerText = data.tokenizer_loaded ? 'Loaded' : 'Not Loaded';
                }
            } catch (error) {
                statusBadge.className = 'status-badge degraded';
                statusText.innerText = 'API Offline';
                generateBtn.setAttribute('disabled', 'true');
            }
        }

        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            if (tab === 'play') {
                event.target.classList.add('active');
                document.getElementById('tab-play').classList.add('active');
            } else {
                event.target.classList.add('active');
                document.getElementById('tab-guide').classList.add('active');
            }
        }

        async function generateText() {
            const prompt = document.getElementById('prompt').value;
            const temp = parseFloat(document.getElementById('temperature').value);
            const tokens = parseInt(document.getElementById('max-tokens').value);
            const topKVal = parseInt(document.getElementById('top-k').value);
            
            const outputDiv = document.getElementById('output');
            const btnSpinner = document.getElementById('btn-spinner');
            const btnText = document.getElementById('btn-text');
            const generateBtn = document.getElementById('generate-btn');
            const metricBanner = document.getElementById('metric-banner');
            
            outputDiv.innerHTML = '';
            outputDiv.classList.add('loading');
            btnSpinner.style.display = 'block';
            btnText.innerText = 'Generating...';
            generateBtn.setAttribute('disabled', 'true');
            metricBanner.style.display = 'none';
            
            const payload = {
                prompt: prompt,
                max_new_tokens: tokens,
                temperature: temp
            };
            if (topKVal > 0) {
                payload.top_k = topKVal;
            }

            const startTime = performance.now();
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                const latency = Math.round(performance.now() - startTime);
                
                if (response.ok) {
                    // Simulated typing/stream effect
                    outputDiv.classList.remove('loading');
                    const text = data.generated_text;
                    let i = 0;
                    
                    function typeEffect() {
                        if (i < text.length) {
                            outputDiv.innerText += text.charAt(i);
                            i++;
                            outputDiv.scrollTop = outputDiv.scrollHeight;
                            setTimeout(typeEffect, 5); // 5ms per char
                        } else {
                            // Enable inputs
                            generateBtn.removeAttribute('disabled');
                            btnSpinner.style.display = 'none';
                            btnText.innerText = 'Generate Completion';
                            
                            // Display metrics
                            document.getElementById('metric-tokens').innerText = tokens;
                            document.getElementById('metric-time').innerText = latency + ' ms';
                            metricBanner.style.display = 'flex';
                        }
                    }
                    typeEffect();
                    
                } else {
                    throw new Error(data.detail || 'Failed to generate text.');
                }
            } catch (error) {
                outputDiv.classList.remove('loading');
                outputDiv.innerHTML = `<span style="color: var(--danger-color)">Error: \${error.message}</span>`;
                generateBtn.removeAttribute('disabled');
                btnSpinner.style.display = 'none';
                btnText.innerText = 'Generate Completion';
            }
        }

        // Initialize on load
        window.addEventListener('load', () => {
            checkHealth();
            // Poll health state every 10 seconds
            setInterval(checkHealth, 10000);
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    """Serve the web UI interface for Mini GPT text generation."""
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Retrieve service health status information.

    Returns:
        Dictionary detailing model loading state, computation device backend, and components.
    """
    return {
        "status": "healthy" if global_state["model_loaded"] else "degraded",
        "model_loaded": global_state["model_loaded"],
        "device": global_state["device"],
        "tokenizer_loaded": global_state["tokenizer"] is not None,
    }


@app.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate completed text starting from input prompt.

    Args:
        req: GenerateRequest containing generation parameters.

    Returns:
        GenerateResponse detailing original prompt, generated text, and token count.
    """
    if not global_state["model_loaded"] or global_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure a trained model checkpoint is available.",
        )
    if global_state["tokenizer"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tokenizer has not been loaded successfully.",
        )

    model = cast(GPT, global_state["model"])
    tokenizer = global_state["tokenizer"]
    device = global_state["device"]

    # Use end-of-text special token as default seed prompt
    prompt_text = req.prompt if req.prompt else "<|endoftext|>"

    try:
        # Tokenize input sequence
        encoded_ids = tokenizer.encode(prompt_text, allowed_special=True)
        x = torch.tensor(encoded_ids, dtype=torch.long, device=device).unsqueeze(
            0
        )  # (1, T)

        # Autoregressive next-token prediction loops
        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            )

        # Decode output token sequences back to string
        generated_ids = y[0].tolist()
        output_text = tokenizer.decode(generated_ids)

        return GenerateResponse(
            prompt=req.prompt,
            generated_text=output_text,
            num_tokens=req.max_new_tokens,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during token generation: {e}",
        ) from e
