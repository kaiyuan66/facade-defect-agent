#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

# Repository root (parent of `agent/`), used to resolve relative dataset paths in jsonl.
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_path(path: str | Path) -> Path:
    """Resolve `path` against REPO_ROOT when it is not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


BASE_INSTRUCTION = (
    "You are a senior facade inspection engineer. Analyze the facade defect using paired RGB and thermal images, "
    "bbox geometry and 3D spatial metadata. "
    "You must output two tasks in strict order: (B) structured labels first, (A) explainable analysis second. "
    "First two lines must be exactly: Severity: <MINOR|MODERATE|MAJOR> then DefectType: <label>. "
    "Never use LOW, HIGH, CRITICAL, or UNKNOWN as the severity label. Do not use the word Unknown in DefectType; "
    "if uncertain use Unspecified defect. "
    "Important definitions: X/Y/Z are defect position coordinates relative to the building coordinate origin; "
    "Z_alt is the elevation of the defect center; Area indicates the facade zone where the defect is located; "
    "if numeric area is missing, do not fabricate geometric area values. "
    "Use bbox evidence but do not rely on any provided defect class name as ground-truth."
)


def normalize_bbox_payload(bboxes: Sequence[Dict[str, Any]]) -> str:
    items: List[Dict[str, Any]] = []
    for idx, b in enumerate(bboxes, start=1):
        raw = b.get("bbox", [])
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        items.append(
            {
                "bbox_id": idx,
                "bbox_norm_xywh": [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])],
            }
        )
    return json.dumps(items, ensure_ascii=False)


def build_user_text(row: Dict[str, Any], bbox_json: str) -> str:
    def _na(v: Any) -> str:
        if v is None or v == "":
            return "N/A"
        s = str(v).strip()
        return s if s else "N/A"

    area_measure = (
        row.get("Area_m2")
        or row.get("Area_mm2")
        or row.get("Defect_Area")
        or row.get("area")
        or row.get("area_m2")
    )
    area_measure_s = _na(area_measure)
    did = row.get("Defect_ID") or row.get("defect_id")
    fn = row.get("Full_Image") or row.get("image_id")
    return (
        f"Defect ID: {_na(did)}\n"
        f"Full image filename: {_na(fn)}\n"
        f"Area (facade zone): {_na(row.get('Area'))}\n"
        f"Area value (geometric): {area_measure_s}\n"
        f"Floor: {_na(row.get('Floor'))}\n"
        f"X: {_na(row.get('X'))}\n"
        f"Y: {_na(row.get('Y'))}\n"
        f"Z: {_na(row.get('Z'))}\n"
        f"Z_alt (elevation): {_na(row.get('Z_alt'))}\n"
        f"BBoxes (normalized xywh): {bbox_json}\n\n"
        "Output format (strict order, no text before line 1):\n"
        "Line 1: Severity: <MINOR|MODERATE|MAJOR>\n"
        "Line 2: DefectType: <single label; if uncertain use Unspecified defect>\n"
        "Then Task A — explainable diagnosis (markdown-style sections):\n"
        "- Defect Identification\n"
        "- Risk Level\n"
        "- Cause Analysis\n"
        "- Maintenance Strategy\n"
        "- Uncertainty and Follow-up Actions\n\n"
        "Rules: severity must be only MINOR, MODERATE, or MAJOR (never LOW/HIGH/CRITICAL/UNKNOWN). "
        "Do not use the word Unknown in DefectType."
    )


def build_messages(
    rgb_path: Optional[Path],
    thermal_path: Optional[Path],
    user_text: str,
    system_instruction: str | None = None,
) -> List[Dict[str, Any]]:
    sys_text = system_instruction.strip() if system_instruction else BASE_INSTRUCTION
    user_content: List[Dict[str, str]] = []
    if rgb_path is not None:
        user_content.append({"type": "image", "image": str(rgb_path)})
    if thermal_path is not None:
        user_content.append({"type": "image", "image": str(thermal_path)})
    user_content.append({"type": "text", "text": user_text})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_text}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def qwen3vl_inputs_from_messages(processor, messages: List[Dict[str, Any]], device) -> Tuple[Dict[str, torch.Tensor], int]:
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_paths: List[str] = []
    for m in messages:
        for c in m.get("content", []):
            if c.get("type") == "image":
                image_paths.append(c["image"])
    images = [Image.open(p).convert("RGB") for p in image_paths]
    enc = processor(text=[prompt_text], images=images, return_tensors="pt")
    enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    return enc, prompt_len
