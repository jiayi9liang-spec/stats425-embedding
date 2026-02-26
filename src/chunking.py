from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    text: str


def simple_char_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> List[Chunk]:
    """
    Chunk text by characters with overlap.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_txt = text[start:end].strip()
        if chunk_txt:
            chunks.append(Chunk(chunk_id=f"c{idx}", text=chunk_txt))
            idx += 1
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks