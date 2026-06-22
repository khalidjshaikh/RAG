#!/usr/bin/env python3
"""
RAG Prompt Loop — interactive CLI.

Commands:
  /ingest <path>     — ingest a text file or directory
  /stats             — show vector-store stats
  /save <path>       — persist the vector store
  /load <path>       — load a persisted vector store
  /model <name>      — switch LLM model at runtime
  /help              — show this help
  /exit  or  Ctrl+D  — quit
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from rag import RAGEngine


def main() -> None:
    load_dotenv()

    # ── Config from environment ───────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k = int(os.getenv("TOP_K", "4"))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    store_path = os.getenv("STORE_PATH", "vector_store.json")

    # ── Initialise engine ─────────────────────────────────
    engine = RAGEngine(
        api_key=api_key,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        store_path=store_path,
    )

    console = Console()

    # Print welcome
    console.print()
    console.print(Panel.fit(
        "[bold cyan]RAG Prompt Loop[/]\n"
        f"  Model: {llm_model}  │  Embed: {embedding_model}  │  Chunks: {engine.chunk_count}\n"
        "  Type /help for commands, /exit to quit.",
        border_style="cyan",
    ))
    console.print()

    # ── Prompt loop ───────────────────────────────────────
    while True:
        try:
            query = console.input("[bold green]❯[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        # ── Commands ──────────────────────────────────────
        if query.startswith("/"):
            parts = query.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/exit" or cmd == "/quit":
                engine._save()
                console.print("[yellow]Bye![/]")
                break

            elif cmd == "/help":
                console.print(Markdown(HELP_TEXT))

            elif cmd == "/stats":
                table = Table(title="RAG Stats")
                table.add_column("Key", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("LLM Model", engine.llm_model)
                table.add_row("Embedding Model", engine.embedding_model)
                table.add_row("Chunk Size", str(engine.chunk_size))
                table.add_row("Chunk Overlap", str(engine.chunk_overlap))
                table.add_row("Top-K", str(engine.top_k))
                table.add_row("Similarity Threshold", str(engine.similarity_threshold))
                table.add_row("Store Chunks", str(engine.chunk_count))
                table.add_row("Store Path", str(engine.store_path))
                console.print(table)

            elif cmd == "/ingest":
                if not arg:
                    console.print("[red]Usage: /ingest <file_or_directory>[/]")
                    continue
                if not os.path.exists(arg):
                    console.print(f"[red]Path not found: {arg}[/]")
                    continue
                if os.path.isdir(arg):
                    engine.ingest_directory(arg)
                else:
                    engine.ingest_file(arg)
                console.print(f"[green]✓ Store now has {engine.chunk_count} chunks[/]")

            elif cmd == "/model":
                if not arg:
                    console.print(f"[yellow]Current model: {engine.llm_model}[/]")
                    continue
                engine.llm_model = arg
                console.print(f"[green]✓ Model switched to {arg}[/]")

            elif cmd == "/save":
                path = arg or engine.store_path or "vector_store.json"
                engine.save(path)
                console.print(f"[green]✓ Saved to {path}[/]")

            elif cmd == "/load":
                path = arg or "vector_store.json"
                engine.store = engine.store.__class__.load(path)
                console.print(f"[green]✓ Loaded from {path} ({engine.chunk_count} chunks)[/]")

            elif cmd == "/clear":
                count = engine.chunk_count
                from vector_store import VectorStore
                engine.store = VectorStore()
                console.print(f"[yellow]Cleared {count} chunks[/]")

            else:
                console.print(f"[red]Unknown command: {cmd}. Type /help[/]")
            continue

        # ── Normal query → RAG ─────────────────────────────
        with console.status("[cyan]Thinking…[/]"):
            try:
                result = engine.answer(query)
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                continue

        # Show answer
        answer_md = Markdown(result["answer"])
        console.print(Panel(answer_md, border_style="green", title="Answer"))

        # Show sources (collapsed)
        if result["sources"]:
            src_text = "\n".join(
                f"  [{i+1}] score={s['score']:.3f}  {s['text'][:100]}…"
                for i, s in enumerate(result["sources"])
            )
            console.print(Panel(src_text, border_style="dim", title="Sources", title_align="right"))


HELP_TEXT = """
# Commands

| Command | Description |
|---------|-------------|
| `/ingest <path>` | Ingest a text file or directory of `.txt` files |
| `/stats`         | Show current configuration and store stats |
| `/model <name>`  | Switch LLM model (e.g. `gpt-4o`, `gpt-4o-mini`) |
| `/save [path]`   | Persist the vector store to disk |
| `/load [path]`   | Load a persisted vector store from disk |
| `/clear`         | Clear all chunks from the vector store |
| `/help`          | Show this help |
| `/exit`          | Exit the prompt loop |

Just type a question to get a RAG answer from your ingested documents.
"""


if __name__ == "__main__":
    main()
