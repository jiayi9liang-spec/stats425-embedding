from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from src.chunking import simple_char_chunk
from src.embedder_guwenbert import GuwenBERTEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ethanyt/guwenbert-base")
    parser.add_argument("--out_dir", type=str, default="outputs/indexes")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = Path(args.corpus_path).read_text(encoding="utf-8")
    chunks = simple_char_chunk(text, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise ValueError("No chunks produced. Is corpus empty?")

    embedder = GuwenBERTEmbedder(args.model_name)
    chunk_texts = [c.text for c in chunks]
    X = embedder.embed(chunk_texts)

    safe_model = args.model_name.replace("/", "__")
    emb_path = out_dir / f"emb_{safe_model}.npy"
    meta_path = out_dir / f"meta_{safe_model}.json"

    np.save(emb_path, X)

    meta = {
        "model_name": args.model_name,
        "corpus_path": str(args.corpus_path),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "num_chunks": len(chunks),
        "chunks": [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved embeddings: {emb_path}")
    print(f"[OK] Saved metadata:  {meta_path}")
    print(f"[OK] Shape: {X.shape}")


if __name__ == "__main__":
    main()