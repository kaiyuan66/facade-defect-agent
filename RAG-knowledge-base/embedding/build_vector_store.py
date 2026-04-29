#!/usr/bin/env python3
"""Build a local vector store from PDF textbooks using Qwen3-Embedding-0.6B."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

_RAG_KB_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TEXTBOOK_DIR = _RAG_KB_ROOT / "textbook"
_DEFAULT_OUTPUT_DIR = _RAG_KB_ROOT / "embedding/store"
_DEFAULT_MODEL = os.environ.get("FACADE_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    page: int
    text: str


def read_pdf_pages(pdf_path: Path) -> Iterable[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            yield page_idx, text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def collect_chunks(textbook_dir: Path, chunk_size: int, overlap: int) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    pdf_files = sorted(textbook_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        for page, page_text in read_pdf_pages(pdf_path):
            for idx, piece in enumerate(chunk_text(page_text, chunk_size, overlap)):
                records.append(
                    ChunkRecord(
                        chunk_id=f"{pdf_path.stem}-p{page}-c{idx}",
                        source_file=pdf_path.name,
                        page=page,
                        text=piece,
                    )
                )
    return records


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_store(
    textbook_dir: Path,
    output_dir: Path,
    model_path: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_chunks(textbook_dir, chunk_size, overlap)
    if not records:
        raise RuntimeError(f"No text chunks extracted from {textbook_dir}")

    model = SentenceTransformer(model_path)
    texts = [r.text for r in records]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    vectors = normalize(vectors)

    np.save(output_dir / "embeddings.npy", vectors)
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    config = {
        "embedding_model": model_path,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(records),
        "dim": int(vectors.shape[1]),
    }
    (output_dir / "store_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local vector store for textbooks.")
    parser.add_argument(
        "--textbook_dir",
        type=Path,
        default=_DEFAULT_TEXTBOOK_DIR,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=_DEFAULT_MODEL,
        help="HuggingFace model id or local path; env FACADE_EMBEDDING_MODEL overrides default.",
    )
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_store(
        textbook_dir=args.textbook_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
