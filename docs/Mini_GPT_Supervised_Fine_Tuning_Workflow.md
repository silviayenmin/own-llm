# Mini GPT -- Supervised Fine-Tuning (SFT) Workflow

## Goal

Transform the pretrained Mini GPT into a chatbot by continuing training
on instruction-response data.

## Overview

``` text
Raw Text
    ↓
Pretraining (Next Token Prediction)
    ↓
base_model.pt
    ↓
Instruction Dataset
    ↓
Supervised Fine-Tuning
    ↓
chat_model.pt
```

## 1. Pretrain First

Train the model on general text.

Output:

``` text
best_model.pt
tokenizer.json
```

## 2. Prepare Instruction Dataset

Store instruction/response pairs.

Example JSON (`data/instruct/qa_dataset.json`):

``` json
[
  {"prompt": "Explain AI", "response": "Artificial Intelligence is the simulation of human intelligence by machines."}
]
```

Format and pad each pair into an independent discrete sequence ending with `<|endoftext|>`:

``` text
Prompt: Explain AI
Response: Artificial Intelligence is the simulation of human intelligence by machines.<|endoftext|>
```

## 3. Tokenization and Padding

Use the **same tokenizer** used during pretraining, but pad each sequence individually to exactly `block_size + 1` so that absolute positional embeddings start at 0 for every prompt. Save as `instruct_train.pt`.

## 4. Load the Pretrained Model

Instead of random initialization:

``` python
model = GPT(config)
```

Load the pretrained checkpoint:

``` python
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

## 5. Continue Training

The training loop is almost identical to pretraining, but loads from `InstructDataset`.

``` python
for x, y in dataloader:
    logits, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
```

Changes compared to pretraining:

  Pretraining                  Fine-Tuning
  ---------------------------- ---------------------------------
  Continuous text              Padded discrete QA sequences
  Large learning rate          Smaller learning rate
  Many epochs                  Fewer epochs (or higher if dataset is tiny)
  Random weights               Pretrained weights

## 6. Save the New Model

``` text
instruct_model.pt
```

## Project Structure

``` text
project/
│
├── training/
│   ├── train.py
│   ├── finetune.py
│   ├── finetune_dataset.py
│   └── dataset.py (InstructDataset)
│
├── data/
│   ├── raw/
│   └── instruct/qa_dataset.json
│
├── experiments/checkpoints/
│   ├── best_model.pt
│   └── instruct_model.pt
```

## Fine-Tuning Pipeline

``` text
Instruction Dataset
        ↓
Format and Pad Sequences (finetune_dataset.py)
        ↓
Save instruct_train.pt
        ↓
Load best_model.pt
        ↓
Continue Training (finetune.py)
        ↓
Save instruct_model.pt
        ↓
Generate Responses via API
```

## Future Roadmap

1.  Base GPT (Pretraining)
2.  Supervised Fine-Tuning (SFT)
3.  Preference Optimization (DPO/RLHF)
4.  Tool Calling
5.  Retrieval-Augmented Generation (RAG)
6.  Memory
7.  AI Agent Capabilities

## Key Takeaway

The model is already a **generative language model** after pretraining.
Fine-tuning does **not** make it generative; it teaches the model to
specialize in behaviors such as chatting, following instructions,
coding, or answering questions.
