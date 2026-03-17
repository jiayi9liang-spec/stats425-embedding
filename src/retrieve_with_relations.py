from expand_query import expand_query
import numpy as np


def retrieve(question, model, chunk_embeddings, chunks, top_k=5):

    query_embedding = model.embed([question])[0]
    expanded_query = expand_query(question)

    scores = np.dot(chunk_embeddings, query_embedding)

    # relation boost
    expanded_query = expand_query(question)

    for i, chunk in enumerate(chunks):
        for token in expanded_query.split():
            if token in chunk:
                scores[i] += 0.05 

    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in top_idx:
        results.append({
            "chunk": chunks[i],
            "score": float(scores[i])
        })

    return results