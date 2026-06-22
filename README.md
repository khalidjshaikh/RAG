# RAG Prompt Loop

A **Retrieval-Augmented Generation** system backed by OpenAI with an interactive prompt loop. Ask questions against your own documents — the system retrieves the most relevant chunks and generates grounded answers.

## Features

- **Ingest** text files or entire directories — automatic chunking with configurable size and overlap
- **Retrieve** the top-k most relevant chunks via cosine similarity on OpenAI embeddings
- **Generate** answers with GPT, grounded in your document context
- **Interactive CLI** with rich output, slash commands, and persistent storage
- **Flexible configuration** via environment variables or runtime commands

## Quick Start

### 1. Setup

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest documents

In the prompt loop, ingest a text file or a directory of `.txt` files:

```bash
python3 main.py
```

```
❯ /ingest ./my_documents
```

Or ingest individual files:

```
❯ /ingest ./report.txt
```

### 4. Ask questions

Just type your question:

```
❯ What are the key findings from the report?
```

The system will:
1. Embed your query
2. Find the most similar chunks in the store
3. Send them as context to the LLM along with your question
4. Display the answer and source snippets

## Configuration

All settings are optional — defaults work out of the box. Configure via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key **(required)** |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for creating embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | Model for generating answers |
| `LLM_TEMPERATURE` | `0.7` | LLM temperature (0.0–2.0) |
| `LLM_MAX_TOKENS` | `1024` | Max tokens in generated answer |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks (chars) |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.0` | Minimum similarity score (0–1) |
| `STORE_PATH` | `vector_store.json` | Path to persist the vector store |

Runtime overrides are also available (e.g. `/model gpt-4o`).

## Commands

| Command | Description |
|---|---|
| `/ingest <path>` | Ingest a `.txt` file or directory |
| `/stats` | Show config and store statistics |
| `/model <name>` | Switch LLM model (e.g. `gpt-4o`) |
| `/save [path]` | Persist vector store to disk |
| `/load [path]` | Restore a saved vector store |
| `/clear` | Remove all chunks from the store |
| `/help` | Show available commands |
| `/exit` | Quit the prompt loop |

## Project Layout

```
.
├── main.py           # Interactive CLI prompt loop
├── rag.py            # RAG engine: embed → retrieve → generate
├── vector_store.py   # In-memory vector store with cosine similarity
├── document.py       # Text loader and chunker
├── requirements.txt  # Python dependencies
└── .env.example      # Configuration template
```

## Architecture

```
 User query
     │
     ▼
  [Embed] ── OpenAI text-embedding-3-small ──→ query vector
     │
     ▼
  [Vector Store] ── cosine similarity search ──→ top-k chunks
     │
     ▼
  [Generate] ── GPT with context prompt ──→ grounded answer
     │
     ▼
  [Display] ── rendered markdown + source citations
```

The vector store is in-memory (NumPy) and optionally persisted to a JSON file. Chunking splits on sentence boundaries with configurable overlap to preserve context across chunk boundaries.

## Requirements

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- Dependencies: `openai`, `numpy`, `rich`, `python-dotenv`
