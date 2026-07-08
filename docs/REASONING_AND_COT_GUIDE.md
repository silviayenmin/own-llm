# Mini GPT Chain-of-Thought (CoT) Reasoning Guide

This guide details how to implement the **Chain-of-Thought (CoT) Reasoning** technique on your custom Mini GPT model. This formatting structure forces the model to write out its logical thinking steps before delivering a final answer, transforming it from a simple Q&A model into a reasoning-based assistant.

---

## 1. How Chain-of-Thought Works

Standard fine-tuning teaches a model to jump directly to the answer. For example:
*   **Prompt:** `What is 12 + 15?`
*   **Response:** `27.`

If a small model makes a mistake on the first token, it cannot correct itself. 

**Chain-of-Thought (CoT)** forces the model to generate a series of intermediate reasoning steps *before* outputting the final answer:
*   **Prompt:** `What is 12 + 15?`
*   **Response:** `Let's think. First, we split the numbers: 12 is 10 + 2, and 15 is 10 + 5. Next, we add the tens: 10 + 10 = 20. Then, we add the ones: 2 + 5 = 7. Finally, we add the results: 20 + 7 = 27. So, the answer is 27.`

Because the model predicts word-by-word, each reasoning word it prints becomes part of its context (working memory), helping it calculate the final correct answer!

---

## 2. Step-by-Step Implementation Plan

To enable this on your model, we will perform the following steps:

### Step 1: Create a Reasoning Dataset Generator
We will create a script `scripts/generate_reasoning_dataset.py` to write a dataset of 1,000+ conversational pairs. Every response in this dataset will follow a strict reasoning template:
```text
Response: Let's think. [Step 1]... [Step 2]... So, the answer is [Final Response].
```

### Step 2: Backup Current Files
We will back up the current dataset and active model weights so that we can easily roll back if needed:
*   Rename `data/instruct/qa_dataset.json` $\rightarrow$ `data/instruct/qa_dataset_general.json`
*   Rename `experiments/checkpoints/instruct_model.pt` $\rightarrow$ `experiments/checkpoints/instruct_model_general.pt`

### Step 3: Run the Preprocessor
Run the dataset formatter to prepare the reasoning tokens:
```powershell
python -m training.finetune_dataset --config configs/config_huge.yaml
```

### Step 4: Execute Fine-Tuning
Fine-tune the pretrained model on the reasoning dataset for 5 epochs:
```powershell
python -m training.finetune --config configs/config_huge.yaml
```

### Step 5: Test the Model
Run queries and observe how the model starts with `"Let's think."` and writes out its reasoning before answering:
```powershell
python generate.py --checkpoint experiments/checkpoints/instruct_model.pt --prompt "Prompt: What is the capital of Japan?\nResponse:"
```

---

## 3. CoT Dataset Templates

Here is how our new dataset responses will look across different domains:

### A. General Knowledge
*   **Prompt:** `What is the capital of Japan?`
*   **Response:** `Let's think. Japan is an island nation located in East Asia. Its largest city, which serves as the seat of the government and the Emperor, is Tokyo. So, the answer is Tokyo.`

### B. Logical Riddles
*   **Prompt:** `What gets wetter the more it dries?`
*   **Response:** `Let's think. When we dry our body, we use an object to absorb water. That object is a towel. As the towel absorbs more water, it becomes wetter while our body becomes drier. So, the answer is a towel.`

### C. Simple Programming
*   **Prompt:** `How do I print a string in Python?`
*   **Response:** `Let's think. In Python, printing output to the console is done using a built-in output function. That function is print(). To print a string, we pass the text inside quotes to the function, like print("Hello"). So, the answer is to use the print() function.`

### D. Chat & Identity
*   **Prompt:** `What is your name?`
*   **Response:** `Let's think. I am a small language model trained to help you write and chat. My creator named me Mini GPT. So, the answer is my name is Mini GPT.`
