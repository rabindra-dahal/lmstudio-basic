from huggingface_hub import hf_hub_download
import os

# Pointing to the lightweight 135M SmolLM2 model repository
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
LOCAL_DIR = os.path.abspath("./local_model")

files_to_download = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt"
]

print(f"Downloading lightweight SmolLM2-135M straight into {LOCAL_DIR}...")
try:
    for filename in files_to_download:
        print(f"Fetching: {filename}...")
        hf_hub_download(
            repo_id=REPO_ID, 
            filename=filename, 
            local_dir=LOCAL_DIR,
            token=False
        )
    print("\n Successfully packed on local disk! You can now run the backend server using this local model without any internet connection.")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
