import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from retrieve_with_relations import retrieve
import numpy as np
import json

# Load metadata
with open("outputs/indexes/meta_ethanyt__guwenbert-base.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# Extract chunk texts
chunks = [c["text"] for c in meta["chunks"]]

# Load embeddings
chunk_embeddings = np.load("outputs/indexes/emb_ethanyt__guwenbert-base.npy")

# Load your embedding model
from src.embedder_guwenbert import GuwenBERTEmbedder

model = GuwenBERTEmbedder()

question = "林黛玉住在哪里"

results = retrieve(
    question,
    model,
    chunk_embeddings,
    chunks,
    top_k=5
)

for r in results:
    print("\n--- RESULT ---\n")
    print(r["chunk"])