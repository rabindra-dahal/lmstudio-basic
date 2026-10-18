import os
import sys

# ==============================================================================
# WARNING SILENCERS
# ==============================================================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# ==============================================================================

import torch
import psutil
import json
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="LM Studio Portable RAG Backend")

# 1. Load Local Language Model
MODEL_PATH = os.path.abspath("./local_model")
print(f"Loading local LLM from: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float32, device_map="auto", local_files_only=True
)

# 2. Load Local Embedding Model (Fast, un-gated, lightweight ~120MB)
print("Initializing embedding engine for text math vectoring...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. RAG Storage Arrays & FAISS Compilation
DOCS_DIR = os.path.abspath("./knowledge_base")
chunks_text_database = []
faiss_index = None

def build_vector_database():
    global faiss_index, chunks_text_database
    chunks_text_database = []
    
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print("⚠️ 'knowledge_base' directory is empty. Operating in standard chat mode.")
        faiss_index = None
        return

    print("📖 Processing documents inside knowledge base...")
    raw_text = ""
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, file), "r", encoding="utf-8") as f:
                raw_text += f.read() + "\n"

    if not raw_text.strip():
        faiss_index = None
        return

    # Slice text up into overlapping paragraphs (500 characters each)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_text_database = text_splitter.split_text(raw_text)

    # Encode text chunks into math vector numbers
    embeddings = embed_model.encode(chunks_text_database, convert_to_numpy=True)
    
    # Initialize FAISS Index (384 dimensions matches the MiniLM embedding width output)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print(f"✅ Vector database built successfully! Indexed {len(chunks_text_database)} document chunks.")

# Trigger vector build on server boot
build_vector_database()

# --- Diagnostic Route for Frontend Dashboard ---
@app.get("/")
async def root_health_check():
    process = psutil.Process(os.getpid())
    ram_used_mb = process.memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "active_backend_engine": f"PyTorch RAG Engine (FAISS Local) [Indexed Chunks: {len(chunks_text_database)}]",
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
        user_query = request.message
        context_retrieved = ""

        # RAG Search Layer: Query vector extraction if index exists
        if faiss_index is not None and len(chunks_text_database) > 0:
            query_vector = embed_model.encode([user_query], convert_to_numpy=True)
            # Find top 2 closest matching text pieces
            distances, indices = faiss_index.search(query_vector, k=2)
            
            retrieved_pieces = []
            for idx in indices[0]:
                if idx != -1 and idx < len(chunks_text_database):
                    retrieved_pieces.append(chunks_text_database[idx])
            
            if retrieved_pieces:
                context_retrieved = "\n".join(retrieved_pieces)

        # Inject retrieved documentation context straight into the prompt layout
        augmented_system_prompt = request.system_prompt
        if context_retrieved:
            augmented_system_prompt += (
                f"\n\nUse the following verified local reference context to answer the user:\n"
                f"\"\"\"\n{context_retrieved}\n\"\"\"\n"
                f"If the answer cannot be found in the context, use your general knowledge but mention it."
            )

        messages = [
            {"role": "system", "content": augmented_system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        tokenized_chat = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        
        if isinstance(tokenized_chat, dict) or hasattr(tokenized_chat, "data"):
            input_ids = tokenized_chat["input_ids"].to(model.device)
        else:
            input_ids = tokenized_chat.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=request.max_tokens,
            do_sample=True if request.temperature > 0 else False,
            temperature=max(request.temperature, 0.01),
        )

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
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
