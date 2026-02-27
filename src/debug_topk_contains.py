from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.embedder_guwenbert import GuwenBERTEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ethanyt/guwenbert-base")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--answer", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    X = np.load(args.emb_path)
    meta = json.loads(Path(args.meta_path).read_text(encoding="utf-8"))
    chunks = meta["chunks"]

    embedder = GuwenBERTEmbedder(args.model_name)
    q = embedder.embed([args.question])[0]

    scores = X @ q
    topk = np.argsort(-scores)[: args.k]

    print("\n=== DEBUG TOP-K ===")
    print("question:", args.question)
    print("answer  :", args.answer)

    for rank, i in enumerate(topk, start=1):
        txt = chunks[i]["text"]
        found = args.answer in txt
        print(f"\n[{rank}] score={float(scores[i]):.4f} chunk_id={chunks[i]['chunk_id']} contains_answer={found}")
        # print a longer preview so you can visually confirm
        preview = txt.replace("\n", " ")
        print(preview[:600])

    print("\nDone.")


if __name__ == "__main__":
    main()