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
base_model.pt
tokenizer.json
```

## 2. Prepare Instruction Dataset

Store instruction/response pairs.

Example JSON:

``` json
{
  "instruction": "Explain AI",
  "response": "Artificial Intelligence is the simulation of human intelligence by machines."
}
```

Convert every sample into plain text:

``` text
### User:
Explain AI

### Assistant:
Artificial Intelligence is the simulation of human intelligence by machines.
```

## 3. Tokenization

Use the **same tokenizer** used during pretraining.

``` text
Conversation
      ↓
Tokenizer
      ↓
Token IDs
```

## 4. Load the Pretrained Model

Instead of random initialization:

``` python
model = GPT(config)
```

Load the pretrained checkpoint:

``` python
model.load_state_dict(torch.load("base_model.pt"))
```

## 5. Continue Training

The training loop is almost identical to pretraining.

``` python
for batch in dataloader:
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
```

Changes compared to pretraining:

  Pretraining                  Fine-Tuning
  ---------------------------- ---------------------------------
  Wikipedia, Books, Articles   Conversations, QA, Instructions
  Large learning rate          Smaller learning rate
  Many epochs                  Fewer epochs
  Random weights               Pretrained weights

## 6. Save the New Model

``` text
chat_model.pt
```

## Project Structure

``` text
project/
│
├── pretrain.py
├── finetune.py
├── trainer.py
├── tokenizer.py
├── model/
│
├── datasets/
│   ├── wikipedia.txt
│   ├── books.txt
│   └── conversations.json
│
├── checkpoints/
│   ├── base_model.pt
│   └── chat_model.pt
```

## Fine-Tuning Pipeline

``` text
Instruction Dataset
        ↓
Format as Chat
        ↓
Tokenize
        ↓
Load base_model.pt
        ↓
Continue Training
        ↓
Save chat_model.pt
        ↓
Generate Responses
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
