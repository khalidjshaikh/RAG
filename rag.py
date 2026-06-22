"""
RAG Core — embed, store, retrieve, and generate answers via OpenAI.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from openai import OpenAI

from document import load_directory
from vector_store import Chunk, VectorStore


class RAGEngine:
    """Orchestrates embedding → retrieval → generation."""

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        store_path: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Vector store (restore from disk if available)
        self.store_path = store_path
        self.store = VectorStore.load(store_path) if store_path else VectorStore()

    # ── Embedding ──────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Get an embedding vector for a single text string."""
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        """Batch-embed multiple texts."""
        if not texts:
            return []
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        # Ensure we return in the same order
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [np.array(d.embedding, dtype=np.float32) for d in sorted_data]

    # ── Ingestion ──────────────────────────────────────────────

    def ingest_text(self, text: str, metadata: Optional[dict] = None) -> None:
        """Chunk, embed, and store a single text string."""
        from document import chunk_text

        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return
        embeddings = self.embed_many(chunks)
        for chunk_text_val, emb in zip(chunks, embeddings):
            self.store.add(Chunk(text=chunk_text_val, embedding=emb, metadata=metadata or {}))

    def ingest_file(self, file_path: str) -> None:
        """Load a text file, chunk it, embed it, and store."""
        from document import load_text, chunk_text

        text = load_text(file_path)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            print(f"  [rag] No chunks from {file_path}")
            return
        embeddings = self.embed_many(chunks)
        for chunk_text_val, emb in zip(chunks, embeddings):
            self.store.add(
                Chunk(
                    text=chunk_text_val,
                    embedding=emb,
                    metadata={"source": os.path.basename(file_path)},
                )
            )
        print(f"  [rag] Ingested {file_path} -> {len(chunks)} chunks")

    def ingest_directory(self, dir_path: str, glob_pattern: str = "*.txt") -> None:
        """Load all matching files in a directory, chunk, embed, and store."""
        chunks = load_directory(dir_path, self.chunk_size, self.chunk_overlap, glob_pattern)
        if not chunks:
            return
        texts = [c[1] for c in chunks]
        embeddings = self.embed_many(texts)
        for (fname, chunk_text_val), emb in zip(chunks, embeddings):
            self.store.add(
                Chunk(
                    text=chunk_text_val,
                    embedding=emb,
                    metadata={"source": fname},
                )
            )
        self._save()
        print(f"  [rag] Directory ingested: {len(chunks)} chunks from {dir_path}")

    # ── Retrieval ──────────────────────────────────────────────

    def retrieve(self, query: str) -> List[tuple[Chunk, float]]:
        """Embed the query and return the top-k most similar chunks."""
        query_emb = self.embed(query)
        return self.store.search(query_emb, self.top_k, self.similarity_threshold)

    # ── Generation ─────────────────────────────────────────────

    def generate(self, query: str, context_chunks: List[tuple[Chunk, float]]) -> str:
        """Build a prompt with context and ask the LLM to answer."""
        if context_chunks:
            context = "\n\n---\n\n".join(c.text for c, _ in context_chunks)
        else:
            context = "(No relevant documents found.)"

        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question based on the "
            "provided context. If the context doesn't contain enough information to answer "
            "the question, say so clearly — do not make up information. "
            "Cite the source context where possible."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    # ── Full RAG round-trip ────────────────────────────────────

    def answer(self, query: str) -> dict:
        """Embed → retrieve → generate. Returns result dict with sources."""
        context_chunks = self.retrieve(query)
        answer_text = self.generate(query, context_chunks)
        sources = [
            {
                "text": c.text[:150],
                "score": round(s, 4),
                "metadata": c.metadata,
            }
            for c, s in context_chunks
        ]
        return {"query": query, "answer": answer_text, "sources": sources}

    # ── Persistence ────────────────────────────────────────────

    def _save(self) -> None:
        if self.store_path:
            self.store.save(self.store_path)

    def save(self, path: str) -> None:
        self.store.save(path)

    @property
    def chunk_count(self) -> int:
        return self.store.size
