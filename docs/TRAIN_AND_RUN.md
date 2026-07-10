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
python -m venv myenv

# 2. Activate the virtual environment
# On Windows (PowerShell):
myenv\Scripts\Activate.ps1
# On Windows (CMD):
.\myenv\Scripts\Activate.ps1
# On Linux/macOS:
source myenv/bin/activate

# 3. Upgrade pip and install core packages
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> PyTorch will install with CPU support by default. If you have a compatible NVIDIA GPU, install PyTorch with CUDA support to accelerate training and generation speed.

---

## 2. Train the Tokenizer

Mini GPT supports BPE (Byte-Pair Encoding) or simple character-level tokenizers. A custom tokenizer must be trained on your raw input corpus before the model can interpret text.

To train the tokenizer, run it as a module using the `-m` flag to ensure internal imports resolve correctly. It will automatically load the tokenizer settings (like type and vocabulary size) from your `configs/config.yaml` file:

```bash
# Train the tokenizer using settings from config.yaml
python -m tokenizer.train_tokenizer --config configs/config.yaml
```

### CLI Parameters:
* `--config`: Path to the YAML configuration file containing model, training, and data settings (e.g., `configs/config.yaml`).

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

## 5. Instruction Fine-Tuning (Instruct Mode)

Once the base model is trained, it only knows how to generate continuous text. To make it answer questions like an assistant (ChatGPT style), we must fine-tune it on a Q&A dataset.

### 1. Preprocess the Instruct Dataset
Convert the JSON Q&A pairs (`data/instruct/qa_dataset.json`) into padded token sequences:
```bash
python -m training.finetune_dataset --config configs/config.yaml
```

### 2. Run the Fine-Tuning Script
Train the pre-trained base model on the instruct data. This will save a new `instruct_model.pt`:
```bash
python -m training.finetune --config configs/config.yaml
```

---

## 6. Run the API and Web UI Playground

We have integrated a premium, glassmorphic dark-mode Web UI directly inside the FastAPI web server. 

### Launch the FastAPI Server
Run the FastAPI application using Uvicorn. To use the newly fine-tuned instruct model, set the environment variable:

```bash
# Activate Uvicorn server in reload mode with Instruct Model
$env:MODEL_CHECKPOINT_PATH="experiments/checkpoints/instruct_model.pt"; python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

> [!NOTE]
> The server automatically searches for your model checkpoint at `experiments/checkpoints/best_model.pt` and your tokenizer at `data/tokenizer/`. You can customize these paths using environment variables:
> `MODEL_CHECKPOINT_PATH` and `TOKENIZER_DIR`.

### Open the Interactive UI Playground
Once the server is running, open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

#### Web UI Key Features:
* **Interactive Parameters Slider**: Dynamically adjust generation characteristics (Temperature, Max New Tokens, Top-K Sampling).
* **Instruct Formatting Checkbox**: Automatically wrap your prompt in `Prompt: {text}\nResponse: ` so the model knows to answer. **(Important: Check this when using instruct_model.pt!)**
* **Live Health Dashboard**: Real-time status indicators checking API online state, calculation device backend (CPU/CUDA), and loaded status of the checkpoint/tokenizer.
* **Instant Seed Chips**: Click quick prompt templates to instantly seed generation.
* **Simulated Typing Stream Effect**: Tokens are generated on the server and printed out sequentially in the terminal area with smooth typewriter animations.
* **In-App Integration Guide**: A native tab detailing key commands.

---

## 7. Run the Verification Tests

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

---

## 8. Scaling to "Huge Parameter" Model (100M+ Params)

To move beyond the basic character-level Mini GPT and train a GPT-2 equivalent model (~124M parameters) on a large dataset:

### 1. Download the Huge Dataset
We use a 500MB chunk of the `TinyStories` dataset from HuggingFace.
```bash
python scripts/download_dataset.py --samples 2500000
```
*(Note: Requires the `datasets` package to be installed via `pip install datasets`)*

### 2. Train the BPE Tokenizer
Character tokenization won't work efficiently for huge models. Train the custom Byte-Pair Encoding (BPE) Tokenizer:
```bash
python scripts/train_bpe.py --vocab_size 4096
```

### 3. Preprocess the Massive Corpus
Convert the 500MB text file into `.bin` tensors (this uses `np.memmap` so it won't crash your RAM):
```bash
python -m training.dataset --config configs/config_huge.yaml
```

### 4. Train the Huge Model
Launch the massive training run. This is configured to maximize a 16GB GPU (like the RTX 4060 Ti). 
```bash
python -m training.train --config configs/config_huge.yaml
```
*(Warning: This process will take many hours. Checkpoints will save periodically to `experiments/checkpoints/best_model.pt`)*
