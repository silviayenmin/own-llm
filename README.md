# Mini GPT - Proof of Concept (PoC)

A clean, production-quality, modular implementation of a decoder-only GPT model from scratch using PyTorch. 

This repository implements all transformer building blocks manually (no `nn.Transformer` or `nn.MultiheadAttention`) using fundamental PyTorch tensor operations, designed to build a 10M–50M parameter model for text generation PoC.

## Directory Structure

```
mini-gpt/
├── data/                  # Datasets & Tokenizer outputs
│   ├── raw/               # Raw input corpora
│   ├── processed/         # Tokenized/preprocessed dataset segments
│   └── tokenizer/         # Trained tokenizer parameters/model files
├── configs/               # YAML configuration files separating setup from logic
├── docs/                  # Design documentation & mathematical notes
├── experiments/           # Training logs, checkpoints, Tensorboard runs
├── model/                 # Core Transformer / GPT implementation
│   ├── config.py          # Model configuration schemas
│   ├── embedding.py       # Custom Token & Positional embeddings
│   ├── attention.py       # Custom Multi-Head Attention layer
│   ├── feedforward.py     # Custom FeedForward MLP network
│   ├── layernorm.py       # Custom Layer Normalization
│   ├── transformer_block.py # Combined Transformer block layer
│   └── gpt.py             # Assembled GPT decoder-only model
├── tokenizer/             # Custom tokenizer logic
│   ├── train_tokenizer.py # Tokenizer training scripts
│   └── tokenizer.py       # Encoder/Decoder class implementation
├── training/              # Training harness and datasets
│   ├── dataset.py         # PyTorch dataset wrappers
│   ├── trainer.py         # Custom trainer loop (epochs, metrics, checkpoints)
│   ├── loss.py            # Loss functions
│   ├── optimizer.py       # Optimizer definitions
│   ├── scheduler.py       # Learning rate scheduler
│   └── train.py           # Training execution script
├── inference/             # Text generation logic
│   ├── generate.py        # Token generation pipeline
│   └── sampling.py        # Temperature, Top-K, Top-P sampling
├── evaluation/            # Evaluation scripts
│   └── metrics.py         # PPL and perplexity computation
├── api/                   # Deployment layers
│   └── main.py            # FastAPI text generation endpoint
├── tests/                 # Unit tests (Pytest)
├── docker/                # Deployment container configuration
├── pyproject.toml         # Packaging and build settings
└── requirements.txt       # Direct pip dependencies
```

## Getting Started

### 1. Prerequisites
- Python 3.11+
- PyTorch (with CUDA support if GPUs are available)

### 2. Installation
To set up locally:
```bash
python -m venv myenv
# On Windows:
myenv\Scripts\activate
# On Linux/macOS:
source myenv/bin/activate

pip install -r requirements.txt
```

### 3. Development Workflow
We are developing the components in order:
1. Project setup (Completed)
2. Configuration system
3. Tokenizer
4. Dataset loader
5. Embedding layer
6. Positional embeddings
7. Layer Normalization
8. Multi-Head Self-Attention
9. Feed Forward Network
10. Transformer Block
11. GPT model
12. Training loop
13. Text generation
14. Evaluation
15. FastAPI inference
16. Docker deployment

## Documentation & Interactive Playground

To help you get started and understand the project's internal mechanics, we provide the following guides:
* 🚀 **[Quick Start Commands Sheet](docs/QUICKSTART.md)**: A concise list of the exact terminal commands to copy, paste, and run sequentially to train the model and start the UI.
* ⚡ **[GPU Training Guide](docs/GPU_GUIDE.md)**: A step-by-step walkthrough for setting up PyTorch with CUDA and running optimized, high-speed training on an NVIDIA GPU.
* 🗺️ **[Architectural & Implementation Workflow Mindmap](docs/WORKFLOW.md)**: Details the internal layout of the model, detailed tensor shapes/dataflows, chronological implementation timeline, and component descriptions.
* 🌌 **[Train, Run with UI, and Test Guide](docs/TRAIN_AND_RUN.md)**: An in-depth step-by-step walkthrough on how to train a custom BPE/Character tokenizer, preprocess text data, run GPT model training, serve the interactive Web UI playground, and run verification tests.
