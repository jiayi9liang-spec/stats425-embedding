from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.embedder_guwenbert import GuwenBERTEmbedder


def load_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--emb_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ethanyt/guwenbert-base")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    X = np.load(args.emb_path)
    meta = json.loads(Path(args.meta_path).read_text(encoding="utf-8"))
    chunks = meta["chunks"]

    qa = load_jsonl(args.qa_path)
    embedder = GuwenBERTEmbedder(args.model_name)

    hits = 0
    for row in qa:
        q = row["question"]
        gold = row.get("answer", "")
        aliases = row.get("answer_aliases", [])

        qv = embedder.embed([q])[0]
        scores = X @ qv
        topk = np.argsort(-scores)[: args.k]

        found = False
        for i in topk:
            txt = chunks[i]["text"]
            if (gold and gold in txt) or any(a in txt for a in aliases):
                found = True
                break

        hits += int(found)
        print(f"{row.get('id','')} hit@{args.k}={found}")

    print(f"\nHit@{args.k} = {hits}/{len(qa)} = {hits/len(qa):.3f}")


if __name__ == "__main__":
    main()