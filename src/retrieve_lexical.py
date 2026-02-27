from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.chunking import simple_char_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    text = Path(args.corpus_path).read_text(encoding="utf-8")
    chunks = simple_char_chunk(text, chunk_size=args.chunk_size, overlap=args.overlap)

    # simple scoring = count of query words appearing
    terms = [t for t in args.query.strip().split() if t]
    scored = []
    for c in chunks:
        score = sum(1 for t in terms if t in c.text)
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: args.k]

    print("\n=== LEXICAL TOP RESULTS ===")
    print("query:", args.query)
    for rank, (s, c) in enumerate(top, start=1):
        print(f"\n[{rank}] score={s} chunk_id={c.chunk_id}")
        print(c.text[:600].replace("\n", " "), "...")
    print("\nDone.")


if __name__ == "__main__":
    main()