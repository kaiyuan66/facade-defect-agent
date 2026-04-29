"""Simple local vector store load/retrieve helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    source_file: str
    page: int
    text: str


class LocalVectorStore:
    def __init__(self, store_dir: Path, embedding_model_path: str):
        self.store_dir = Path(store_dir)
        self.embedding_model_path = embedding_model_path
        self.embeddings = np.load(self.store_dir / "embeddings.npy")
        self.metadata = []
        with (self.store_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        self.model = SentenceTransformer(self.embedding_model_path)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        scores = self.embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        results: List[RetrievedChunk] = []
        for i in idx:
            m = self.metadata[int(i)]
            results.append(
                RetrievedChunk(
                    score=float(scores[i]),
                    chunk_id=m["chunk_id"],
                    source_file=m["source_file"],
                    page=int(m["page"]),
                    text=m["text"],
                )
            )
        return results
