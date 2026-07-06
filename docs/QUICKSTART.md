# 🚀 Mini GPT Quick Start Commands

This is a concise, step-by-step command reference sheet. Run these commands in order in your project root terminal to set up, train, and launch the application.

---

## 🛠️ Step 1: Environment Setup

```powershell
# 1. Create a virtual environment
python -m venv myenv

# 2. Activate the virtual environment
# On Windows (PowerShell):
myenv\Scripts\Activate.ps1
# On Windows (CMD):
myenv\Scripts\activate.bat
# On Linux/macOS:
source myenv/bin/activate

# 3. Upgrade pip and install dependencies
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Step 2: Train the Tokenizer

Trains a custom Byte-Pair Encoding (BPE) or Character tokenizer on the raw text dataset. Running python with the `-m` flag ensures Python resolves local directory modules like `model` correctly.

```powershell
python -m tokenizer.train_tokenizer --config configs/config.yaml
```

---

## 📦 Step 3: Preprocess the Dataset

Tokenizes the raw corpus and saves it into binary arrays (`train.bin`, `val.bin`) for efficient training load.

```powershell
python -m training.dataset --config configs/config.yaml
```

---

## 🏋️ Step 4: Train the Base GPT Model

Trains the custom GPT transformer network using the configuration hyperparameters.

```powershell
# Option A: Quick Tiny CPU training run (Recommended for testing setup, ~1 minute)
python -m training.train --config configs/config_tiny.yaml

# Option B: Standard training run (Larger network, requires CUDA or patience on CPU)
python -m training.train --config configs/config.yaml
```

---

## 🤖 Step 5: Instruction Fine-Tuning (Optional)

Fine-tune the base model to act as an assistant and answer questions in a Q&A format.

```powershell
# 1. Process and pad the Q&A dataset
python -m training.finetune_dataset --config configs/config.yaml

# 2. Train the Instruct model
python -m training.finetune --config configs/config.yaml
```

---

## 🌐 Step 6: Start the Web UI Playground

Launches the FastAPI server serving the interactive glassmorphic UI playground.

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

👉 Now open your browser and go to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Step 7: Run Tests (Optional Validation)

Runs all 56 mathematical and functional tests to confirm the system's structural integrity.

```powershell
python -m pytest
```
