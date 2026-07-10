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
    # Attach our custom reasoning layers onto the pre-trained base model
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

            # Format input using Qwen-2.5 chat template
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize and push to GPU
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response skipping template formatting
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"\nMini GPT (1.5B): {response.strip()}\n" + "-"*50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError generating response: {e}\n")

if __name__ == "__main__":
    chat()
