import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

app = FastAPI(title="LM Studio GGUF Backend Clone")

# Change these lines to use Qwen2.5 instead
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q8_0.gguf"


print("Checking and downloading GGUF model file...")

# FORCE download directly into your local project folder to avoid path bugs
local_dir_path = os.path.abspath("./models")
model_path = hf_hub_download(
    repo_id=MODEL_REPO, 
    filename=MODEL_FILE, 
    local_dir=local_dir_path,
    token=False
)

# Convert path explicitly to standard absolute format
model_path = os.path.abspath(model_path)
print(f"Loading GGUF model from: {model_path}")

try:
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,       
        n_threads=4,      
        verbose=False
    )
except Exception as e:
    print(f"\n[ERROR] llama-cpp failed to load model again. Internal message: {e}")
    raise e

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

        response_stream = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        )

        def text_generator():
            for chunk in response_stream:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}) # Adjusted to index 0 safely
                    token = delta.get("content", "")
                    if token:
                        yield json.dumps({"delta": token}) + "\n"

        return StreamingResponse(text_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root_health_check():
#     return {"status": "healthy", "message": "LM Studio Backend is running!"}

import psutil  # You may need to run: pip install psutil

@app.get("/")
async def root_health_check():
    # 1. Detect which libraries are loaded into this specific process memory
    globals_dict = globals()
    if "Llama" in globals_dict or "llama_cpp" in globals_dict:
        engine_type = "GGUF Engine (llama-cpp-python) [LIGHTWEIGHT]"
    elif "AutoModelForCausalLM" in globals_dict or "transformers" in globals_dict:
        engine_type = "PyTorch Engine (transformers) [HEAVY RAM]"
    else:
        engine_type = "Unknown / Generic FastAPI Engine"

    # 2. Calculate the exact RAM this backend process is hoarding
    process = psutil.Process(os.getpid())
    ram_used_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "status": "healthy",
        "active_backend_engine": engine_type,
        "backend_ram_usage_mb": f"{ram_used_mb:.2f} MB",
        "running_file_path": os.path.abspath(__file__)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
