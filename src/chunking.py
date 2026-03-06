from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    text: str


def simple_char_chunk(text: str, min_chunk_size: int = 50, overlap: int = 0) -> List[Chunk]:
    """
    Paragraph-based chunking using teammate's logic.
    overlap is kept only for compatibility and is not used.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = text.split("\n")
    chunks: List[Chunk] = []
    current_chunk = ""
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) >= min_chunk_size:
            if current_chunk:
                chunks.append(Chunk(chunk_id=f"c{idx}", text=current_chunk))
                idx += 1
                current_chunk = ""
            chunks.append(Chunk(chunk_id=f"c{idx}", text=para))
            idx += 1
        else:
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para

            if len(current_chunk) >= min_chunk_size:
                chunks.append(Chunk(chunk_id=f"c{idx}", text=current_chunk))
                idx += 1
                current_chunk = ""

    if current_chunk:
        chunks.append(Chunk(chunk_id=f"c{idx}", text=current_chunk))

    return chunks