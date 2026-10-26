# Response: AI Project Details — Khalid Shaikh

**To:** Recruiting / Agentic Bulk initiative review team
**From:** Khalid Shaikh
**Re:** Detailed write-up for AI project — Retrieval-Augmented Generation (RAG) Knowledge Assistant

---

## 1. Project: RAG Prompt Loop — Document Grounded Q&A System

### Purpose of the system
I built an end-to-end **Retrieval-Augmented Generation (RAG) system** that lets a user ask natural-language questions against their own document corpus and receive answers that are grounded in the source text, with citations back to the retrieved passages. The deliverable is an interactive CLI application (`main.py`) plus a reusable Python library (`rag.py`, `vector_store.py`, `document.py`) that can be embedded into other services.

### Business challenge it was designed to solve
Teams accumulate large volumes of unstructured text (reports, reference lists, internal docs) that are slow and error-prone to search by keyword. The challenge was to reduce the time it takes to find a specific fact inside a large corpus and to avoid the hallucinations typical of "chat with your LLM" demos by forcing the model to answer **only from retrieved context**. Concretely, the system had to:
- Ingest arbitrary `.txt` files or whole directories without manual preprocessing.
- Return relevant passages in milliseconds, even as the corpus grew to thousands of chunks.
- Produce answers that cite their sources so a reviewer can verify them.
- Be configurable and operable by non-engineers (model, chunk size, top-k, thresholds via env vars or slash commands).

### How the solution addressed the challenge
I owned the full implementation (design, code, testing, and packaging):

- **Ingestion & chunking** (`document.py`): A loader reads text files/directories and a sentence-boundary-aware chunker splits them into ~512-character overlapping chunks (64-char overlap) so context is preserved across chunk boundaries. Long sentences are hard-split as a fallback.
- **Embeddings** (`rag.py`): Uses OpenAI `text-embedding-3-small` (1536-dim). `embed_many` batches up to 100 texts per API call and preserves result ordering, keeping ingestion cost and latency low.
- **Vector store** (`vector_store.py`): In-memory NumPy store computing cosine similarity via normalized dot product; supports add/search/save/load. Persisted to JSON so a built index can be reused across sessions without re-embedding.
- **Retrieval**: Top-k (default 4) chunks above a configurable similarity threshold, returned sorted by score.
- **Generation** (`rag.py`): Builds a context prompt from retrieved chunks and calls GPT (`gpt-4o-mini` by default, swappable at runtime) with a system prompt that forbids fabricating information and requires citing context. Returns the answer plus structured source snippets (text preview, score, metadata).
- **CLI / operability** (`main.py`): Rich-rendered interactive loop with slash commands (`/ingest`, `/stats`, `/model`, `/save`, `/load`, `/clear`, `/help`), runtime config overrides, and graceful error handling.

### Quantifiable results and metrics (measured)

Corpus used for validation: a reference document corpus ingested into **2,491 chunks**.

| Metric | Value |
|---|---|
| Chunks indexed | 2,491 |
| Embedding dimension | 1,536 (text-embedding-3-small) |
| Chunk size / overlap | 512 chars / 64 chars |
| Top-k retrieved per query | 4 |
| Chunking throughput | 2,491 chunks in 0.05 s |
| Batch embedding throughput | ~125 texts/sec (100 texts in 0.80 s) |
| Single-query embedding latency | ~0.99 s |
| Vector search latency (p95, 2,491 chunks) | 14.3 ms (avg 8.6 ms) |
| End-to-end answer latency (embed + retrieve + generate) | ~1.95 s |
| Generation latency (gpt-4o-mini) | ~1.50 s |
| Persistent store size | 2,491 chunks in `vector_store.json` (~78 MB), reloadable |

**Outcome:** A user question that previously required manually scanning a large document set now resolves in under ~2 seconds with a grounded answer and 4 cited source passages, eliminating ungrounded LLM responses by construction (the model is instructed to say when context is insufficient).

### My role and technical ownership
- Sole author of all modules (`main.py`, `rag.py`, `vector_store.py`, `document.py`).
- Designed the retrieval pipeline, chunking strategy, batching, persistence format, and CLI UX.
- Implemented cosine-similarity search, batched embedding with order preservation, and source-attributed answer generation.
- Configured and validated against OpenAI APIs; managed API key/config via environment variables.

### Stack
Python 3.10+, OpenAI API (embeddings + chat), NumPy, Rich, python-dotenv.

---

*Happy to walk through the codebase, run a live demo, or share the repository.*