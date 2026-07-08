import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

# Add project root to PYTHONPATH so we can import model and tokenizer packages
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from model.config import GPTConfig
from model.gpt import GPT
from generate import load_tokenizer

app = FastAPI(title="Mini GPT Chat Interface")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
pad_token_id = None

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    top_k: int = None
    max_new_tokens: int = 150

@app.on_event("startup")
def startup_event():
    global model, tokenizer, device, pad_token_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading API backend on device: {device}...")
    
    checkpoint_path = os.path.join(project_root, "experiments", "checkpoints", "instruct_model.pt")
    tokenizer_dir = os.path.join(project_root, "data", "tokenizer")
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Instruct model checkpoint not found at {checkpoint_path}")
        
    # Load model and config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)
    config.training.device = device
    
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_dir)
    pad_token_id = tokenizer.special_stoi.get("<|endoftext|>", 256)
    print("Model and Tokenizer loaded successfully!")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global model, tokenizer, device, pad_token_id
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is loading or not initialized.")
        
    try:
        # Wrap prompt in standard Instruct fine-tuning template
        formatted_prompt = f"Prompt: {request.prompt}\nResponse:"
        encoded_ids = tokenizer.encode(formatted_prompt, allowed_special=True)
        
        x = torch.tensor(encoded_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )
            
        generated_ids = y[0].tolist()
        prompt_len = len(encoded_ids)
        new_tokens = generated_ids[prompt_len:]
        
        # Stop at the first stop token (pad_token_id)
        if pad_token_id in new_tokens:
            first_pad_idx = new_tokens.index(pad_token_id)
            generated_ids = generated_ids[:prompt_len + first_pad_idx]
            
        full_output = tokenizer.decode(generated_ids)
        
        # Extract the Response segment only (strip the Prompt part)
        response_marker = "Response:"
        if response_marker in full_output:
            response_text = full_output.split(response_marker, 1)[1].strip()
        else:
            response_text = full_output.replace(formatted_prompt, "").strip()
            
        return {"response": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).parent / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
