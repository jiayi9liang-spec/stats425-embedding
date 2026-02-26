from __future__ import annotations
import argparse
import json
from pathlib import Path

import faiss

from embedder_guwenbert import GuwenBERTEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ethanyt/guwenbert-base")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    index = faiss.read_index(args.index_path)
    meta = json.loads(Path(args.meta_path).read_text(encoding="utf-8"))
    chunks = meta["chunks"]

    embedder = GuwenBERTEmbedder(args.model_name)

    qv = embedder.embed([args.question])
    scores, ids = index.search(qv, args.k)

    print("\n=== TOP RESULTS ===")
    for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
        if i < 0:
            continue
        print(f"\n[{rank}] score={float(s):.4f} chunk_id={chunks[i]['chunk_id']}")
        print(chunks[i]["text"][:300], "...")
    print("\nDone.")


if __name__ == "__main__":
    main()