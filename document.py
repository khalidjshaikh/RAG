"""
Document loader — reads text files from a directory and splits them into chunks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from vector_store import Chunk


def load_text(path: str) -> str:
    """Read a single text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters,
    trying to break at sentence boundaries.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split into sentences (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += " " + sent if current else sent
        else:
            if current:
                chunks.append(current)
            # Start new chunk; if the sentence itself is too long, hard-split it
            if len(sent) > chunk_size:
                for i in range(0, len(sent), chunk_size - overlap):
                    chunks.append(sent[i : i + chunk_size])
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)

    # Apply overlap by appending overlap suffix to each chunk
    # (simpler: just keep the chunks as-is with overlap via windowing)
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i < len(chunks) - 1:
                # Append start of next chunk as overlap
                next_start = chunks[i + 1][:overlap]
                c = c + " " + next_start
            overlapped.append(c)
        chunks = overlapped

    return chunks


def load_directory(
    dir_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    glob_pattern: str = "*.txt",
) -> List[tuple[str, str]]:
    """
    Walk a directory, read all matching files, and return (filename, text_chunk) pairs.
    """
    base = Path(dir_path)
    if not base.exists():
        print(f"  [doc] Directory not found: {dir_path}")
        return []

    all_chunks: List[tuple[str, str]] = []
    for fpath in sorted(base.rglob(glob_pattern)):
        text = load_text(str(fpath))
        raw_chunks = chunk_text(text, chunk_size, chunk_overlap)
        for chunk in raw_chunks:
            all_chunks.append((fpath.name, chunk))

    print(f"  [doc] Loaded {len(all_chunks)} chunks from {dir_path} ({glob_pattern})")
    return all_chunks
