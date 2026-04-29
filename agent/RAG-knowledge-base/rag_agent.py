#!/usr/bin/env python3
"""Generic RAG agent entrypoint for textbook QA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from embedding.vector_store import LocalVectorStore

_RAG_KB_ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = _RAG_KB_ROOT / "prompt/system_prompt.md"
COT_PROMPT_PATH = _RAG_KB_ROOT / "prompt/cot_prompt.md"
_DEFAULT_STORE = _RAG_KB_ROOT / "embedding/store"
_DEFAULT_EMBEDDING = os.environ.get("FACADE_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def make_context(chunks) -> str:
    lines = []
    for idx, c in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] source={c.source_file} page={c.page} score={c.score:.4f}\n{c.text}"
        )
    return "\n\n".join(lines)


def ask_with_rag(
    question: str,
    store_dir: Path,
    embedding_model_path: str,
    llm_model: str,
    base_url: str,
    api_key: str,
    top_k: int = 6,
) -> Dict[str, Any]:
    store = LocalVectorStore(store_dir=store_dir, embedding_model_path=embedding_model_path)
    chunks = store.retrieve(question, top_k=top_k)
    context = make_context(chunks)

    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)
    cot_prompt = load_prompt(COT_PROMPT_PATH)

    user_prompt = (
        f"{cot_prompt}\n\n"
        f"[User Question]\n{question}\n\n"
        f"[Retrieved Evidence]\n{context}\n\n"
        "Please provide a structured conclusion grounded in the evidence above and include citation indices."
    )

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": [c.__dict__ for c in chunks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG query over textbook vector store.")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument(
        "--store_dir",
        type=Path,
        default=_DEFAULT_STORE,
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=_DEFAULT_EMBEDDING,
        help="HuggingFace model id or local directory; env FACADE_EMBEDDING_MODEL overrides default.",
    )
    parser.add_argument("--llm_model", type=str, default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--top_k", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = ask_with_rag(
        question=args.question,
        store_dir=args.store_dir,
        embedding_model_path=args.embedding_model_path,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key=args.api_key,
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
