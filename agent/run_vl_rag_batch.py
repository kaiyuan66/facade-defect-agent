#!/usr/bin/env python3
"""Run batch multimodal defect explanation with local vLLM + RAG."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent/RAG-knowledge-base"))
from embedding.vector_store import LocalVectorStore  # noqa: E402

_DEFAULT_LABELS = ROOT / "dataset/image/labels.jsonl"
_DEFAULT_IMAGE_ROOT = ROOT / "dataset/image/20251003094651"
_DEFAULT_STORE_DIR = ROOT / "agent/RAG-knowledge-base/embedding/store"
_DEFAULT_EMBED_MODEL = os.environ.get("FACADE_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
SYSTEM_PROMPT_PATH = ROOT / "agent/RAG-knowledge-base/prompt/system_prompt.md"
COT_PROMPT_PATH = ROOT / "agent/RAG-knowledge-base/prompt/cot_prompt.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_annotations(path: Path) -> List[dict]:
    """Load labels from JSONL (one object per line) or legacy JSON array file."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix.lower() == ".jsonl":
        out: List[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported labels format in {path}")


def data_url(img_path: Path, max_side: int = 768, quality: int = 85) -> str:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def collect_images(root: Path, suffix: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in root.rglob(f"*{suffix}.jpg"):
        out[p.name] = p
    return out


def build_rag_context(store: LocalVectorStore, query: str, top_k: int = 3, max_chars: int = 500) -> str:
    chunks = store.retrieve(query, top_k=top_k)
    lines = []
    for idx, c in enumerate(chunks, start=1):
        clipped = c.text[:max_chars]
        lines.append(f"[{idx}] {c.source_file} p.{c.page} score={c.score:.4f}\n{clipped}")
    return "\n\n".join(lines)


def compact_ann(anns: List[dict], max_items: int = 12) -> List[dict]:
    out = []
    for a in anns[:max_items]:
        bbox = [round(float(x), 4) for x in a["bbox"]]
        out.append({"label": a["label"], "bbox_norm_xywh": bbox})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch run VL+RAG for first N locations.")
    p.add_argument("--num_locations", type=int, default=100)
    p.add_argument("--base_url", type=str, default="http://127.0.0.1:18002/v1")
    p.add_argument("--api_key", type=str, default="EMPTY")
    p.add_argument("--model", type=str, default="Qwen3-VL-8B-Instruct")
    p.add_argument(
        "--label_json",
        type=Path,
        default=_DEFAULT_LABELS,
        help="Annotations: JSONL (one record per line) or JSON array (.json). Default: dataset/image/labels.jsonl.",
    )
    p.add_argument("--image_root", type=Path, default=_DEFAULT_IMAGE_ROOT, help="Root with visual/ and thermal/ subtrees.")
    p.add_argument("--store_dir", type=Path, default=_DEFAULT_STORE_DIR, help="RAG vector store directory.")
    p.add_argument(
        "--embedding_model",
        type=str,
        default=_DEFAULT_EMBED_MODEL,
        help="Embedding model for retrieval (HuggingFace id or local path). Override with env FACADE_EMBEDDING_MODEL.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "result/vl_rag_first100",
    )
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--skip_existing", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.image_root
    label_json = args.label_json

    visual_map = collect_images(image_root / "visual", "_V")
    thermal_map = collect_images(image_root / "thermal", "_T")
    labels = load_annotations(label_json)
    ann_by_file: Dict[str, List[dict]] = defaultdict(list)
    for item in labels:
        ann_by_file[item["filename"]].append(item)

    # Use all available visual images as candidate locations.
    # If a location has no annotation in JSON, it will be treated as unlabeled.
    ordered_files: List[str] = sorted(visual_map.keys())

    store = LocalVectorStore(store_dir=args.store_dir, embedding_model_path=args.embedding_model)
    system_prompt = read_text(SYSTEM_PROMPT_PATH)
    cot_prompt = read_text(COT_PROMPT_PATH)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    results = []
    existing_txt = {
        p.name.replace(".txt", ".jpg")
        for p in args.out_dir.glob("*.txt")
        if p.name.endswith(".txt")
    }
    existing_count = len(existing_txt) if args.skip_existing else 0
    processed = 0
    total_processed = existing_count
    skipped = 0
    for visual_name in ordered_files:
        if total_processed >= args.num_locations:
            break
        if args.skip_existing and visual_name in existing_txt:
            continue

        thermal_name = visual_name.replace("_V.jpg", "_T.jpg")
        visual_path = visual_map.get(visual_name)
        thermal_path = thermal_map.get(thermal_name)
        if not visual_path or not thermal_path:
            skipped += 1
            continue

        anns = ann_by_file.get(visual_name, [])
        labels_in_img = sorted({a["label"] for a in anns}) if anns else ["Unknown"]
        rag_query = (
            "Facade defect diagnosis and repair recommendations for "
            f"labels: {', '.join(labels_in_img)}. "
            "Focus on code-compliant risk grading, probable causes, and maintenance strategy."
        )
        rag_context = build_rag_context(store, rag_query, top_k=args.top_k, max_chars=500)
        ann_payload = compact_ann(anns, max_items=12) if anns else []
        label_count = defaultdict(int)
        for a in anns:
            label_count[a["label"]] += 1
        if not label_count:
            label_count["Unknown"] = 0

        user_text = (
            f"{cot_prompt}\n\n"
            "[Task]\n"
            "Analyze one facade location with paired RGB and thermal images plus detection annotations.\n"
            "Provide an explainable diagnosis that follows engineering code logic.\n\n"
            f"[Image Pair ID]\n{visual_name}\n\n"
            f"[Label Summary]\n{json.dumps(label_count, ensure_ascii=False)}\n\n"
            "[Annotations]\n"
            f"{json.dumps(ann_payload, ensure_ascii=False)}\n\n"
            "[Retrieved RAG Evidence]\n"
            f"{rag_context}\n\n"
            "Output sections: Defect Identification, Risk Level, Cause Analysis, Maintenance Strategy, "
            "Uncertainty and Follow-up Actions, Evidence Citations."
        )

        resp = client.chat.completions.create(
            model=args.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url(visual_path)}},
                        {"type": "image_url", "image_url": {"url": data_url(thermal_path)}},
                    ],
                },
            ],
            max_tokens=1200,
        )
        answer = resp.choices[0].message.content

        out_item = {
            "image_id": visual_name,
            "visual_path": str(visual_path),
            "thermal_path": str(thermal_path),
            "labels": labels_in_img,
            "num_boxes": len(anns),
            "analysis_text": answer,
        }
        results.append(out_item)
        (args.out_dir / f"{visual_name.replace('.jpg', '.txt')}").write_text(
            answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        processed += 1
        total_processed += 1
        print(f"processed_new={processed} total={total_processed} file={visual_name}")

    (args.out_dir / "results.json").write_text(
        json.dumps(
            {
                "processed_new": processed,
                "existing_count": existing_count,
                "processed_total": total_processed,
                "skipped": skipped,
                "requested": args.num_locations,
                "items": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "processed_new": processed,
                "existing_count": existing_count,
                "processed_total": total_processed,
                "skipped": skipped,
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
