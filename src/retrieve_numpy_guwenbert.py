from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.embedder_guwenbert import GuwenBERTEmbedder


def retrieve(
    question: str,
    emb_path: str,
    meta_path: str,
    model_name: str = "ethanyt/guwenbert-base",
    k: int = 5,
):
    X = np.load(emb_path)  # (n, d), normalized
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    chunks = meta["chunks"]

    embedder = GuwenBERTEmbedder(model_name)
    q = embedder.embed([question])[0]  # (d,), normalized

    # cosine similarity = dot product since normalized
    scores = X @ q  # (n,)
    topk_idx = np.argsort(-scores)[:k]

    results = []
    for rank, i in enumerate(topk_idx, start=1):
        results.append(
            {
                "rank": rank,
                "chunk_id": chunks[i]["chunk_id"],
                "retrieval_score": float(scores[i]),
                "text": chunks[i]["text"],
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ethanyt/guwenbert-base")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve(
        question=args.question,
        emb_path=args.emb_path,
        meta_path=args.meta_path,
        model_name=args.model_name,
        k=args.k,
    )

    print("\n=== TOP RESULTS ===")
    for r in results:
        print(f"\n[{r['rank']}] score={r['retrieval_score']:.4f} chunk_id={r['chunk_id']}")
        print(r["text"][:300], "...")
    print("\nDone.")


if __name__ == "__main__":
    main()
