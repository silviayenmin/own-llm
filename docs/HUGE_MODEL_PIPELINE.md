# 🗂️ Huge Model Pipeline: Pretraining & Fine-tuning

This document provides a clean, step-by-step guide to train the **Huge GPT Model (124M parameters)** from scratch on the **TinyStories** dataset, and subsequently fine-tune it on the custom Q&A dataset.

---

> [!WARNING]
> **Warning on Checkpoints:**
> Training the Huge Model will overwrite the checkpoints in the `experiments/checkpoints/` directory (e.g., `best_model.pt`). If you wish to keep your old Mini GPT model checkpoints, please move them to a backup folder before proceeding.

---

## 🛠️ Step-by-Step Commands

### Step 0: Activate Virtual Environment
Make sure you are in the project root directory and your environment is active in your terminal:
```powershell
.\myenv\Scripts\Activate.ps1
```
*(You should see `(myenv)` prefixing your command line).*

---

## 1. 🏗️ Pretraining Phase (Creating the Base Model)

### Step 1.1: Train the BPE Tokenizer
This trains the Byte-Pair Encoding tokenizer using a 0.5MB sample of the dataset to learn the vocabulary of words:
```powershell
$env:PYTHONPATH="."; python scripts/train_bpe.py --vocab_size 4096 --chunk_size_mb 0.5
```
* **Verify:** Ensure `data/tokenizer/tokenizer.json` is created.

### Step 1.2: Preprocess the Large Dataset
This processes the first 30MB of `huge_corpus.txt`, tokenizes the words into integers using the cached BPE tokenizer, and splits them into memory-mapped train/val binary files:
```powershell
python -m training.dataset --config configs/config_huge.yaml
```
* **Verify:** Ensure `data/processed/train.bin` and `data/processed/val.bin` are generated.

### Step 1.3: Train the Base Model (Pretraining)
This trains the 124M parameter model from scratch on the preprocessed TinyStories dataset. This is a heavy step; it will utilize your RTX 4060 Ti GPU:
```powershell
$env:PYTHONPATH="."; python -m training.train --config configs/config_huge.yaml
```
* **Verify:** Ensure the training completes successfully and the final weights are saved to `experiments/checkpoints/best_model.pt`.

---

## 2. 🎯 Fine-tuning Phase (Instruction Tuning)

Once the base model is fully trained, we can fine-tune it on your custom instructions (e.g., Q&A questions).

### Step 2.1: Preprocess the Instruct Dataset
This reads your Q&A dataset at `data/instruct/qa_dataset.json`, formats the prompts with the BPE tokenizer, formats them, and saves the binary training tensors:
```powershell
python -m training.finetune_dataset --config configs/config_huge.yaml
```
* **Verify:** Ensure `data/processed/instruct_train.pt` and `data/processed/instruct_val.pt` are generated.

### Step 2.2: Fine-tune the Model
This loads the pretrained base model weights from `experiments/checkpoints/best_model.pt` and updates them on the instruction dataset.
```powershell
python -m training.finetune --config configs/config_huge.yaml
```
* **Verify:** Ensure the training completes and the fine-tuned weights are saved to `experiments/checkpoints/instruct_model.pt`.

---

## 💬 3. Run and Generate Text

To test your models, run the following generation commands:

### To test the Base Model (Generates story text):
```powershell
python generate.py --checkpoint experiments/checkpoints/best_model.pt --prompt "Once upon a time, there was a little bird"
```

### To test the Instruct Model (Answers your prompts):
```powershell
python generate.py --checkpoint experiments/checkpoints/instruct_model.pt --prompt "Prompt: What is your name?\nResponse:"
```
