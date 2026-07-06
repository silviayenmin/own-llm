# 🌌 Mini GPT: Train, Run with UI, and Test Guide

This guide provides a comprehensive step-by-step walkthrough for setting up the environment, training the custom tokenizer, preprocessing datasets, training the Mini GPT model, running the interactive Web UI, and executing tests.

---

## 📋 Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Step 1: Train the Tokenizer](#step-2-train-the-tokenizer)
3. [Step 3: Preprocess the Raw Text Corpus](#step-3-preprocess-the-raw-text-corpus)
4. [Step 4: Train the GPT Model](#step-4-train-the-gpt-model)
5. [Step 5: Run the API and Web UI Playground](#step-5-run-the-api-and-web-ui-playground)
6. [Step 6: Run the Verification Tests](#step-6-run-the-verification-tests)

---

## 1. Prerequisites & Environment Setup

Ensure you have Python 3.11+ installed. Execute the following commands in the project root to configure the virtual environment and install dependencies:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# 3. Upgrade pip and install core packages
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> PyTorch will install with CPU support by default. If you have a compatible NVIDIA GPU, install PyTorch with CUDA support to accelerate training and generation speed.

---

## 2. Train the Tokenizer

Mini GPT supports BPE (Byte-Pair Encoding) or simple character-level tokenizers. A custom tokenizer must be trained on your raw input corpus before the model can interpret text.

To train the tokenizer, run it as a module using the `-m` flag to ensure internal imports resolve correctly:

```bash
# Train a BPE tokenizer (Recommended)
python -m tokenizer.train_tokenizer --type bpe --vocab_size 320 --raw_data_path data/raw/input.txt --tokenizer_dir data/tokenizer

# Train a Character-level tokenizer (Simpler/Alternative)
python -m tokenizer.train_tokenizer --type char --raw_data_path data/raw/input.txt --tokenizer_dir data/tokenizer
```

### CLI Parameters:
* `--type`: The tokenizer architecture, either `bpe` or `char`.
* `--vocab_size`: Target vocabulary size (only applies to BPE merges).
* `--raw_data_path`: Path to the raw text corpus (defaults to `data/raw/input.txt`).
* `--tokenizer_dir`: Destination directory to output the generated `tokenizer.json` configuration.

---

## 3. Preprocess the Raw Text Corpus

Once the tokenizer is trained and saved to `data/tokenizer`, you need to encode the raw text into binary formats (`train.bin` and `val.bin`) for memory-efficient loading during model training.

Run the preprocessing module:

```bash
python -m training.dataset --tokenizer_dir data/tokenizer
```

> [!TIP]
> This command tokenizes the text from `data/raw/input.txt` and splits it into a training set (`train.bin`, 90% default split) and validation set (`val.bin`, 10%) inside the `data/processed/` directory.

---

## 4. Train the GPT Model

Now you are ready to train the neural network. We provide two default configurations:
1. `configs/config_tiny.yaml`: A small 2-layer model configured for CPU validation and quick verification.
2. `configs/config.yaml`: The default configuration for longer-running experiments.

### Run a Quick Tiny CPU Training Run
To verify the entire setup is working without waiting hours for training:

```bash
python -m training.train --config configs/config_tiny.yaml
```

### Run Standard Model Training
```bash
python -m training.train --config configs/config.yaml
```

The script will:
* Load configuration files
* Initialize the model architectures manually (custom embeddings, layer norms, and multi-head attention blocks)
* Run epochs, log loss to standard out, and save checkpoints to `experiments/checkpoints/best_model.pt`.

---

## 5. Run the API and Web UI Playground

We have integrated a premium, glassmorphic dark-mode Web UI directly inside the FastAPI web server. 

### Launch the FastAPI Server
Run the FastAPI application using Uvicorn:

```bash
# Activate Uvicorn server in reload mode
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

> [!NOTE]
> The server automatically searches for your model checkpoint at `experiments/checkpoints/best_model.pt` and your tokenizer at `data/tokenizer/`. You can customize these paths using environment variables:
> `MODEL_CHECKPOINT_PATH` and `TOKENIZER_DIR`.

### Open the Interactive UI Playground
Once the server is running, open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

#### Web UI Key Features:
* **Interactive Parameters Slider**: Dynamically adjust generation characteristics (Temperature, Max New Tokens, Top-K Sampling).
* **Live Health Dashboard**: Real-time status indicators checking API online state, calculation device backend (CPU/CUDA), and loaded status of the checkpoint/tokenizer.
* **Instant Seed Chips**: Click quick prompt templates (like `First Citizen:`) to instantly seed generation.
* **Simulated Typing Stream Effect**: Tokens are generated on the server and printed out sequentially in the terminal area with smooth typewriter animations.
* **In-App Integration Guide**: A native tab detailing key commands.

---

## 6. Run the Verification Tests

A comprehensive unit test suite is provided to ensure all components behave identically to mathematical specs.

Execute the following command in the project root:

```bash
python -m pytest
```

This runs 56 unit tests asserting core correctness, including:
* Embedding shapes, causal masks, layer norm scaling, and forward pass accuracy.
* Dataset creation and serialization logic.
* Text generation sampling (greedy vs. stochastic).
* API health checks, Web UI routes, and POST inference responses.
