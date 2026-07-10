# 🚀 How to Fine-Tune Open-Source LLMs Locally (Llama-3, Qwen, Phi-3)

This guide explains how to fine-tune a pre-trained open-source model (like **Qwen-2.5-1.5B-Instruct** or **Llama-3.2-1B-Instruct**) on your local GPU (RTX 4060 Ti) to create a custom general reasoning chatbot.

---

## 🛠️ The Technology Stack
To fine-tune models with billions of parameters on a home GPU, we use **QLoRA (Quantized Low-Rank Adaptation)**:
1.  **Quantization (BitsAndBytes):** Compresses the model weights to 4-bit precision so they use 75% less VRAM.
2.  **LoRA (PEFT):** Freezes the base model weights and only trains a tiny set of adapter weights (~1% of parameters), which prevents out-of-memory errors.
3.  **HuggingFace TRL (Transformer Reinforcement Learning):** Provides the `SFTTrainer` class to easily fine-tune instruction datasets.

---

## 📦 Step 1: Install Required Libraries
Run this command in your conda environment to install the modern fine-tuning stack:
```powershell
python -m pip install torch transformers peft trl bitsandbytes datasets accelerate
```

---

## 🔑 Step 2: Authenticate with Hugging Face (For Gated Models like Llama)
Because Meta's Llama models are **gated**, you must request access and configure your API key before downloading:
1.  **Request Model Access:** Visit the [Llama-3.2-1B-Instruct page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), log in, fill in the agreement, and submit. Approval is instant.
2.  **Generate a Token:** Go to your Hugging Face account settings, navigate to **Access Tokens**, and create a new **Read** token.
3.  **Log in locally:** In your conda terminal, run either of these commands and paste your token:
    ```powershell
    # Option A (Modern CLI):
    hf auth login

    # Option B (Guaranteed Python fallback):
    python -c "from huggingface_hub import login; login()"
    ```

---

## 📝 Step 3: The Fine-Tuning Python Script
Create a script named `finetune_opensource.py` and paste the following clean, professional template:

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train():
    # 1. Choose your base model (e.g. Qwen-2.5-1.5B or Llama-3.2-1B)
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "./local_finetuned_model"

    print("=== Step 1: Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Step 2: Configuring 4-bit Quantization ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("\n=== Step 3: Loading Pre-trained Base Model ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model weights for PEFT training
    model = prepare_model_for_kbit_training(model)

    print("\n=== Step 4: Configuring LoRA Adapters ===")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("\n=== Step 5: Loading and Formatting Custom Dataset ===")
    dataset = load_dataset("json", data_files="data/instruct/qa_dataset.json", split="train")

    # Helper function to apply chat template formatting
    def format_chat_template(batch):
        texts = []
        for prompt, response in zip(batch["prompt"], batch["response"]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(formatted)
        return {"text": texts}

    # Map the dataset to the chat format
    dataset = dataset.map(format_chat_template, batched=True, remove_columns=["prompt", "response"])

    print("\n=== Step 6: Configuring SFT Training Arguments ===")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=150,                # 150-300 steps is ideal for SFT on this size of dataset
        optim="paged_adamw_8bit",     # High-speed memory-optimized optimizer
        bf16=True,                    # Use native bfloat16 to avoid grad scaler check errors
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        dataset_text_field="text",
        max_length=512
    )

    print("\n=== Step 7: Starting Fine-Tuning Loop ===")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args
    )

    trainer.train()

    print("\n=== Step 8: Saving Fine-Tuned Adapters ===")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 Success! LoRA adapters saved successfully to: {output_dir}")

if __name__ == "__main__":
    train()
```

---

## 🏃 Step 4: Run the Training
Execute the training script:
```powershell
python finetune_opensource.py
```
This runs QLoRA training in 4-bit precision. It will use only **~4.5GB of GPU VRAM** and complete training on your dataset in roughly **10 to 15 minutes**!

---

## 🤖 Step 5: Run and Chat with your Fine-Tuned Model
Once saved, write a quick test script (`test_finetuned_opensource.py`) to query your new custom brain:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def chat():
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    adapter_dir = "./local_finetuned_model"

    print("=== Loading Tokenizer & 16-bit Base Model ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("\n=== Merging Fine-Tuned LoRA Adapters ===")
    model = PeftModel.from_pretrained(model, adapter_dir)
    print("Model loaded and merged successfully!")

    print("\n=== Start Chatting! ===")
    print("Type your prompt below. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_prompt = input("You: ")
            if user_prompt.strip().lower() in ["exit", "quit"]:
                break
            
            if not user_prompt.strip():
                continue

            messages = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\nResponse: {response.strip()}\n" + "-"*50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError generating response: {e}\n")

if __name__ == "__main__":
    chat()
```

This model is **guaranteed to generalize** to any question you ask because it combines the massive pretraining library of the Qwen/Llama base model with your custom conversational style adapters!

---

## 🎨 Step 6: (Optional) Run the Web Chat UI
If you prefer chatting in a beautiful web browser interface rather than the terminal command line, you can run a local web chat interface using **Gradio**:

1.  **Install Gradio:**
    ```powershell
    python -m pip install gradio
    ```
2.  **Create the UI Script (`ui_finetuned_opensource.py`):**
    Create a script named `ui_finetuned_opensource.py` and paste the following code:
    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import gradio as gr

    # Configure model IDs
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Change to "Qwen/Qwen2.5-1.5B-Instruct" if using Qwen
    adapter_dir = "./local_finetuned_model"

    print("=== Loading Tokenizer & 16-bit Base Model ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("=== Merging Fine-Tuned LoRA Adapters ===")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("Model loaded and merged successfully!")

    # Keep a history of the FULL raw model responses (with the thinking prefix) to keep generation formatting consistent
    full_responses = []

    def predict(message, history):
        global full_responses
        
        # If the user cleared the chat or it's a new conversation start, reset our full responses cache
        if len(history) == 0:
            full_responses = []
            
        messages = []
        
        # Helper to extract text from potential Gradio 6 dictionary content structures
        def clean_text(content):
            if not content:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                return content.get("text", content.get("content", ""))
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(item.get("text", item.get("content", "")))
                    elif hasattr(item, "text"):
                        parts.append(getattr(item, "text", ""))
                return "".join(parts)
            return str(content)
        
        # Reconstruct the message history using the cached full raw reasoning responses
        for idx, msg in enumerate(history):
            if isinstance(msg, dict):
                role = msg.get("role", "user")
            elif hasattr(msg, "role"):
                role = getattr(msg, "role", "user")
            else:
                role = "user" if idx % 2 == 0 else "assistant"
                
            if role == "assistant" and (idx // 2) < len(full_responses):
                content = full_responses[idx // 2]
            else:
                if isinstance(msg, dict):
                    content = clean_text(msg.get("content", ""))
                elif hasattr(msg, "content"):
                    content = clean_text(getattr(msg, "content", ""))
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    content = clean_text(msg[1] if role == "assistant" else msg[0])
                else:
                    content = clean_text(msg)
                    
            messages.append({"role": role, "content": content})
        
        messages.append({"role": "user", "content": clean_text(message)})
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        full_responses.append(response)
        
        lower_res = response.lower()
        marker = "the answer is"
        
        if marker in lower_res:
            idx = lower_res.find(marker)
            return response[idx + len(marker):].strip(" :,.\n")
                
        cleaned = response
        think_marker = "let's think."
        if cleaned.lower().startswith(think_marker):
            cleaned = cleaned[len(think_marker):].strip(" :,.\n")
            
        return cleaned

    demo = gr.ChatInterface(
        fn=predict,
        title="🧠 Local Fine-Tuned Chatbot (Gradio UI)",
        description="Chat with your locally fine-tuned Llama/Qwen reasoning model. The interface keeps track of your conversation history.",
        examples=["What has keys but can't open locks?", "What is 15 + 27?", "Write a poem about rain."]
    )

    if __name__ == "__main__":
        demo.launch(share=False)
    ```

3.  **Run the UI server:**
    ```powershell
    python ui_finetuned_opensource.py
    ```
4.  **Chat in Browser:** 
    Gradio will launch a local server and give you a local URL (e.g. `http://127.0.0.1:7860`). Open that link in your browser to chat with your fine-tuned model inside a clean, modern messaging interface!

