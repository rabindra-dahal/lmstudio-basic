# Local LM Studio Playground with RAG Architecture

A fully portable, lightweight, and completely **offline AI playground** built from scratch using **FastAPI** and **Streamlit**. This project mirrors the foundational capabilities of applications like LM Studio by hosting Small Language Models (SLMs) locally, streaming responses token-by-token, and utilizing a local vector database (**FAISS**) for Retrieval-Augmented Generation (RAG) against custom text documents.

Optimized to run smoothly on standard laptops (tested on a **Dell Latitude 3420 with an 11th Gen Intel CPU and 16GB RAM**), it maintains a minimal memory footprint while ensuring 100% conversation privacy.

---

## 🌟 Core Features

- **Decoupled Architecture:** Clean separation between the FastAPI streaming server and the interactive Streamlit user interface.
- **100% Offline & Portable:** All weights, configurations, tokenizers, and vector embeddings are stored locally within the project folder. No internet connection is required after the initial setup.
- **Local RAG Integration:** Implements semantic context injection using **FAISS** (Facebook AI Similarity Search) and **Sentence-Transformers** to chat with documents in real time.
- **Dynamic Configuration Dashboard:** Control parameters like System Prompt guidelines, generation Temperature, and Maximum Tokens directly from the UI.
- **Safety Guardrails:** Real-time **Word & Token Counters** equipped with a dynamic CPU-latency warning box when context limits scale beyond 2,048 tokens.
- **UI Diagnostics:** Real-time backend tracking showing the active engine framework and the exact system RAM memory overhead in megabytes.
- **Session Persistence:** Full chat history persistence layer backing up records instantly to local disk storage (`chat_history.json`).
- **Developer Features:** Includes a "Clear Chat" pipeline, a browser-native "Export Chat to Markdown File" generator, and automated code block syntax highlighting.

---

## 📂 Project Directory Structure

```text
lmstudio-basic/
├── .streamlit/
│   └── config.toml           # UI Theme Configurations (Dark/Light Theme)
├── knowledge_base/
│   └── reference_notes.txt   # Target text files for RAG search injections
├── local_model/              # Packed 135M LLM weights (SmolLM2-135M-Instruct)
├── local_embed_model/        # Packed ~90MB Embedding model weights
├── backend-transformer.py    # FastAPI Server running local PyTorch Inference
├── frontend.py               # Streamlit web interface dashboard canvas
├── chat_history.json         # Local text backup file for persistent messages
└── README.md                 # Project Documentation
```

---

## ⚙️ Installation & Environment Setup

Ensure you have Python 3.10+ installed on your system. Open your terminal inside the project directory and execute the following steps:

### 1. Initialize and Activate Virtual Environment
```bash
python -m venv venv
# On Windows Command Prompt:
venv\Scripts\activate
# On Windows PowerShell:
.\venv\Scripts\activate
```

### 2. Install Hardware & Framework Dependencies
```bash
pip install fastapi uvicorn streamlit requests transformers torch accelerate psutil faiss-cpu sentence-transformers langchain-text-splitters
```

---

## 📥 Gathering the Portable Model Assets

Before executing the localized applications, run these quick downloader scripts to bundle the model architectures straight into your local directory slots.

### Step A: Pack the LLM Engine (`local_model/`)
Create a temporary `download_llm.py` file and run it to fetch the lightweight, high-speed **SmolLM2-135M-Instruct** model:
```python
from huggingface_hub import hf_hub_download
import os

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
LOCAL_DIR = os.path.abspath("./local_model")
files = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]

print("Packing LLM files locally...")
for f in files:
    hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=LOCAL_DIR, token=False)
print("🎉 LLM downloaded successfully!")
```
Run: `python download_llm.py`

### Step B: Pack the Embedding Engine (`local_embed_model/`)
Create a temporary `download_embed.py` script to bundle the **all-MiniLM-L6-v2** semantic processor vector module locally:
```python
from sentence_transformers import SentenceTransformer
import os

LOCAL_DIR = os.path.abspath("./local_embed_model")
print("Packing embedding model files locally...")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save(LOCAL_DIR)
print("🎉 Embedding model downloaded successfully!")
```
Run: `python download_embed.py`

*(Note: You can safely delete both temporary downloader scripts after they finish compiling).*

---

## 🚀 Execution Guide

Always launch your systems from two separate terminal windows inside your virtual environment to ensure proper process separation.

### Step 1: Start the Local FastAPI Backend Server
```bash
python backend-transformer.py
```
The server will initialize your `knowledge_base` text data vectors, bind your C++ thread processors efficiently, and listen silently for streaming requests on `http://127.0.0.1:8000`.

### Step 2: Start the Streamlit Graphical Interface
```bash
streamlit run frontend.py
```
Streamlit will automatically hook into your backend configurations and map out your visual canvas environment inside your browser window at `http://localhost:8501`.

---

## 🌙 UI Theme Overrides (`.streamlit/config.toml`)
To switch the interface background dynamically to match the authentic **LM Studio dark theme layout**, populate your `.streamlit/config.toml` file with this specification profile:

```toml
[theme]
primaryColor = "#ff4b4b"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1f2937"
textColor = "#ffffff"
font = "sans serif"
```

---

## 📝 Performance Tuning & Hardware Notes

- **RAM Benchmarks:** On a 16GB RAM platform environment, this decoupled PyTorch/FAISS layout locks down stably around **~2.1 GB to 2.3 GB of total working memory allocation overhead**, leaving plenty of breathing space for multitasking.
- **Context Boundaries:** While the model natively handles up to 8k context splits, keeping your user interface parameter boundaries set under **512 or 1,024 tokens** prevents processor bottleneck lags on consumer laptop CPUs.
- **Data Security:** No telemetry, metric data, or conversational string fragments are ever transmitted online. All inferences occur purely over your local system memory bus.
