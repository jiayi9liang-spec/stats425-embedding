import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
from retrieve_with_relations import retrieve
from src.embedder_guwenbert import GuwenBERTEmbedder


# load questions
questions = []

with open("data/qa/qa.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        questions.append(json.loads(line))


# load chunks
with open("outputs/indexes/meta_ethanyt__guwenbert-base.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

chunks = [c["text"] for c in meta["chunks"]]


# load embeddings
chunk_embeddings = np.load("outputs/indexes/emb_ethanyt__guwenbert-base.npy")


# load model
model = GuwenBERTEmbedder()


hits = 0
mrr_total = 0
precision_total = 0
num_questions = len(questions)


for q in questions:

    question = q["question"]
    answer = q["answer"]

    results = retrieve(
        question,
        model,
        chunk_embeddings,
        chunks,
        top_k=5
    )

    found = False

    for rank, r in enumerate(results, start=1):

        if answer in r["chunk"]:

            if not found:
                hits += 1
                mrr_total += 1 / rank
                found = True

            precision_total += 1


hit_rate = hits / num_questions
mrr = mrr_total / num_questions
precision = precision_total / (num_questions * 5)


print("\nRelation-Augmented Retrieval Results\n")
print("Hit@5:", hit_rate)
print("MRR:", mrr)
print("Precision@5:", precision)