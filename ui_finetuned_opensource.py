import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# Configure model IDs
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Change this to "Qwen/Qwen2.5-1.5B-Instruct" if switching
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
        # Extract role
        if isinstance(msg, dict):
            role = msg.get("role", "user")
        elif hasattr(msg, "role"):
            role = getattr(msg, "role", "user")
        else:
            role = "user" if idx % 2 == 0 else "assistant"
            
        # Extract content (inject full raw model response for assistant if cached)
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
    
    # Append current user message
    messages.append({"role": "user", "content": clean_text(message)})
    
    # Format prompts with the model's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate output token streams
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    # Save the full raw response (with thinking prefix) to keep subsequent formatting consistent
    full_responses.append(response)
    
    # 1. Search for 'the answer is' case-insensitively (Standard case)
    lower_res = response.lower()
    marker = "the answer is"
    
    if marker in lower_res:
        idx = lower_res.find(marker)
        return response[idx + len(marker):].strip(" :,.\n")
            
    # 2. If marker was not outputted, try to filter out thinking sentences using keyword analysis
    # Split response into individual words/sentences to detect where the actual conversational dialog starts
    cleaned_sentences = []
    reasoning_keywords = ["user", "greeting", "respond", "reply", "think", "logic", "question", "ask", "answer", "well-being"]
    conversational_triggers = ["hello", "hi", "how are you", "fine", "thanks", "thank you", "good morning", "good afternoon"]
    
    # We clean up the response sentences
    raw_sentences = response.split(". ")
    for sentence in raw_sentences:
        clean_s = sentence.strip()
        if not clean_s:
            continue
            
        # Strip the leading "Let's think" or "let's think" if present in the sentence
        if clean_s.lower().startswith("let's think"):
            # If the sentence contains text after "Let's think", clean it, otherwise skip it
            clean_s = clean_s[len("let's think"):].strip(" :,.\n")
            if not clean_s:
                continue
                
        # If the sentence consists only of reasoning patterns, strip it
        is_reasoning = any(kw in clean_s.lower() for kw in reasoning_keywords)
        is_conversational = any(trigger in clean_s.lower() for trigger in conversational_triggers)
        
        # If it's reasoning and NOT a common chat greeting, strip it
        if is_reasoning and not is_conversational:
            continue
            
        # Re-attach clean sentence
        cleaned_sentences.append(clean_s)
        
    if cleaned_sentences:
        return ". ".join(cleaned_sentences).strip(" :,.\n")
        
    # 3. Fallback: if everything was filtered or nothing matched, return raw string without "Let's think."
    cleaned = response
    think_marker = "let's think."
    if cleaned.lower().startswith(think_marker):
        cleaned = cleaned[len(think_marker):].strip(" :,.\n")
        
    return cleaned

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=predict,
    title="🧠 Local Fine-Tuned Chatbot (Gradio UI)",
    description="Chat with your locally fine-tuned Llama/Qwen reasoning model. The interface keeps track of your conversation history.",
    examples=["What has keys but can't open locks?", "What is 15 + 27?", "Write a poem about rain."]
)

if __name__ == "__main__":
    demo.launch(share=False)
