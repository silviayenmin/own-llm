import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train():
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

    print("\n=== Step 3: Loading Pre-trained 1.5B Base Model ===")
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
    # SFTTrainer will automatically wrap the model with get_peft_model using our config below

    print("\n=== Step 5: Loading and Formatting Custom Dataset ===")
    # Load dataset from JSON
    dataset = load_dataset("json", data_files="data/instruct/qa_dataset.json", split="train")

    # Helper function to apply Qwen chat template formatting
    def format_qwen_chat(batch):
        texts = []
        for prompt, response in zip(batch["prompt"], batch["response"]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            # Convert conversations into a single string formatted for Qwen
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(formatted)
        return {"text": texts}

    # Map the dataset to the chat format
    dataset = dataset.map(format_qwen_chat, batched=True, remove_columns=["prompt", "response"])
    print(f"Dataset formatted successfully! First sample text:\n{dataset[0]['text'][:200]}...")

    print("\n=== Step 6: Configuring Training Arguments ===")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=150,                # 150 steps is ideal for SFT on this size of dataset
        optim="paged_adamw_8bit",     # High-speed memory-optimized optimizer
        bf16=True,
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
