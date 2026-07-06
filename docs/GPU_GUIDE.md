# ⚡ GPU Training Guide for Mini GPT

If you have an NVIDIA GPU, you can scale up the Mini GPT model's parameters and train it to produce highly accurate Shakespearean English in just a few minutes. 

Follow these steps to transition from CPU training to GPU training.

---

## Step 1: Install PyTorch with CUDA Support
By default, the setup installs a CPU-only version of PyTorch. To enable GPU acceleration, activate your virtual environment (`.venv`) and reinstall PyTorch with CUDA support:

```powershell
# 1. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 2. Uninstall the CPU-only PyTorch version
pip uninstall torch -y

# 3. Install PyTorch with CUDA 12.1 support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

To verify the installation worked and your GPU is detected:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Step 2: Configure the GPU Profile
Open `configs/config.yaml` and modify it to increase the model's capacity and switch the target device to `"cuda"`. 

Here is the recommended configuration:

```yaml
model:
  vocab_size: 2000
  block_size: 256         # Larger context window (captures longer-range grammar)
  n_layer: 6              # 6 attention layers instead of 4
  n_head: 8               # 8 attention heads instead of 4
  n_embd: 256             # Larger embedding dimension (larger neural memory)
  dropout: 0.1
  bias: false

training:
  batch_size: 64          # Process 64 sequences in parallel on GPU (fast!)
  learning_rate: 1.0e-3
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  epochs: 15              # Train for more epochs to improve spelling
  warmup_iters: 20
  lr_decay_iters: 300
  min_lr: 1.0e-4
  eval_interval: 50       
  eval_iters: 10
  log_interval: 10
  checkpoint_dir: "experiments/checkpoints"
  device: "cuda"          # <--- Target NVIDIA GPU using CUDA
  seed: 1337

data:
  raw_data_path: "data/raw/input.txt"
  processed_dir: "data/processed"
  tokenizer_dir: "data/tokenizer"
  train_split: 0.9
```

---

## Step 3: Run the Training Pipeline
With CUDA active and the config file updated, execute the following commands in order:

### 1. Preprocess the Dataset
Since the `block_size` changed from `128` to `256`, we re-align the dataset mapping:
```powershell
python -m training.dataset --tokenizer_dir data/tokenizer
```

### 2. Train the Model on GPU
Start the training script with unbuffered output (`-u`) to see live metrics:
```powershell
python -u -m training.train --config configs/config.yaml
```
*(On a modern GPU, this will complete 15 epochs in under 2 minutes, and the loss will drop below `4.0`, yielding perfect spelling!)*

### 3. Launch the Web Interface
Start the FastAPI server to load the new GPU-trained checkpoint:
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser and test the prompt generator.
