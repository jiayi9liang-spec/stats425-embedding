from __future__ import annotations
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class GuwenBERTEmbedder:
    """
    Embeddings using GuwenBERT via mean pooling of token embeddings + L2 normalize.
    """

    def __init__(self, model_name: str = "ethanyt/guwenbert-base", device: str | None = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 16, max_length: int = 256) -> np.ndarray:
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)
            token_embeddings = out.last_hidden_state  # (B, T, H)
            attention_mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)

            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)  # (B, H)
            counts = attention_mask.sum(dim=1).clamp(min=1)  # (B, 1)
            mean_pooled = summed / counts  # (B, H)

            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_vecs.append(mean_pooled.cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs)