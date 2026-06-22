"""
Vector Store — in-memory with NumPy cosine similarity.
Supports add, search, save, and load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Chunk:
    """A single document chunk with its embedding and metadata."""
    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            text=d["text"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            metadata=d.get("metadata", {}),
        )


class VectorStore:
    """Simple in-memory vector store with cosine-similarity search."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunk: Chunk) -> None:
        self._chunks.append(chunk)

    def add_many(self, chunks: List[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, top_k: int = 4, threshold: float = 0.0) -> List[tuple[Chunk, float]]:
        """
        Return up to top_k chunks sorted by cosine similarity.
        Only returns results with similarity > threshold.
        """
        if not self._chunks:
            return []

        # Stack all embeddings: (N, D)
        all_embs = np.stack([c.embedding for c in self._chunks])

        # Cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        all_norms = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-12)
        sims = all_norms @ query_norm  # (N,)

        # Filter + sort
        indices = np.argsort(-sims)  # descending
        results = []
        for idx in indices:
            if sims[idx] <= threshold:
                break
            results.append((self._chunks[idx], float(sims[idx])))
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str) -> None:
        """Persist to a JSON file."""
        data = [c.to_dict() for c in self._chunks]
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  [store] Saved {len(data)} chunks to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load from a JSON file created by save()."""
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        store = cls()
        for d in data:
            store._chunks.append(Chunk.from_dict(d))
        print(f"  [store] Loaded {len(store._chunks)} chunks from {path}")
        return store
