import os
import sys

# ==============================================================================
# WARNING SILENCERS: Must run before importing heavy machine learning libraries
# ==============================================================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# ==============================================================================

import torch
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import json

app = FastAPI(title="LM Studio Backend Clone")

# Point to your local project directory folder instead of an online repository ID
MODEL_PATH = os.path.abspath("./local_model")
print(f"Loading local model directly from project folder: {MODEL_PATH}...")

# Force local-only loading to prevent any internet pings or cache lookups
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    dtype=torch.float32, 
    device_map="auto",
    local_files_only=True
)

# --- Diagnostic Route for Frontend Dashboard ---
@app.get("/")
async def root_health_check():
    process = psutil.Process(os.getpid())
    ram_used_mb = process.memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "active_backend_engine": "PyTorch Engine (transformers) [PORTABLE]",
        "backend_ram_usage_mb": f"{ram_used_mb:.2f} MB",
        "running_file_path": os.path.abspath(__file__)
    }

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant."
    temperature: float = 0.7
    max_tokens: int = 512

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ]
        
        # Format the conversational layout array
        tokenized_chat = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        # Extract the raw tensor directly to bypass thread sequence mapping errors
        if isinstance(tokenized_chat, dict) or hasattr(tokenized_chat, "data"):
            input_ids = tokenized_chat["input_ids"].to(model.device)
        else:
            input_ids = tokenized_chat.to(model.device)

        # Setup text stream processor
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=request.max_tokens,
            do_sample=True if request.temperature > 0 else False,
            temperature=max(request.temperature, 0.01),
        )

        # Spin off the generation worker in a non-blocking background thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def text_generator():
            for new_text in streamer:
                if new_text:
                    yield json.dumps({"delta": new_text}) + "\n"

        return StreamingResponse(text_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Silence ONLY the repetitive access logs to keep the console clean from ping spam
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Run server with clear info parameters visible on startup
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
