from sentence_transformers import SentenceTransformer
import os

# Define a local folder path for the embedding model
embed_local_path = os.path.abspath("./local_embed_model")

print(f"Downloading and saving embedding model to: {embed_local_path}")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save(embed_local_path)
print("Embedding model packed locally!")
