#!/usr/bin/env python3
"""Evaluate local base model + LoRA adapter without vLLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from utils.metrics import evaluate_texts
from vl_multimodal_utils import (
    build_messages,
    build_user_text,
    normalize_bbox_payload,
    qwen3vl_inputs_from_messages,
    resolve_data_path,
)


DEFECT_CLASSES = [
    "Crack",
    "Spalling",
    "Sign of Degradation",
    "Glass Anomaly (Spot)",
    "Glass Anomaly (Line)",
    "Debonding",
    "Peeling Paint",
    "Incorrect Installation",
    "Unspecified defect",
]
SEV_CLASSES = ["MINOR", "MODERATE", "MAJOR"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--lora_path", type=str, required=True)
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--out_file", type=Path, required=True)
    p.add_argument("--max_samples", type=int, default=400)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument(
        "--system_prompt_file",
        type=Path,
        default=_REPO_ROOT / "RAG-knowledge-base/prompt/system_prompt.md",
    )
    p.add_argument(
        "--bertscore_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="BERTScore 使用的 HuggingFace 模型 id；设为空字符串则不算 BERTScore",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def pick_defect_label(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("defecttype:"):
            val = s.split(":", 1)[1].strip()
            if not val:
                break
            lowv = val.lower()
            if "unspecified" in lowv:
                return "Unspecified defect"
            for c in DEFECT_CLASSES:
                if c.lower() == lowv or c.lower() in lowv:
                    return c
            return "Unspecified defect"
    low = text.lower()
    for c in DEFECT_CLASSES:
        if c.lower() in low:
            return c
    return "Unspecified defect"


def normalize_severity_gt(raw: str) -> str:
    m = str(raw or "").upper().strip()
    if m in ("MAJOR", "HIGH", "CRITICAL"):
        return "MAJOR"
    if m == "MODERATE":
        return "MODERATE"
    return "MINOR"


def pick_severity(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("severity:"):
            rest = s.split(":", 1)[1].strip().upper().replace(",", " ")
            toks = [t for t in rest.split() if t]
            if toks and toks[0] in SEV_CLASSES:
                return toks[0]
            for k in ("MAJOR", "MODERATE", "MINOR"):
                if k in rest:
                    return k
            return "MINOR"
    up = text.upper()
    for k in ("MAJOR", "MODERATE", "MINOR"):
        if k in up:
            return k
    return "MINOR"


def macro_prf1(y_true: List[str], y_pred: List[str], labels: List[str]) -> dict:
    eps = 1e-12
    p_list, r_list, f_list = [], [], []
    for lb in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lb and p == lb)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lb and p == lb)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lb and p != lb)
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f = 2 * p * r / (p + r + eps)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    return {
        "precision_macro": sum(p_list) / len(p_list),
        "recall_macro": sum(r_list) / len(r_list),
        "f1_macro": sum(f_list) / len(f_list),
    }


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    mt = getattr(cfg, "model_type", "")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    devmap = "auto" if torch.cuda.is_available() else None
    if mt == "qwen3_vl":
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=devmap,
        )
    elif mt in ("deepseek_vl", "deepseek_vl_hybrid"):
        base = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=devmap,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=devmap,
        )
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    rows = load_jsonl(args.test_file)[: args.max_samples]
    system_instruction = args.system_prompt_file.read_text(encoding="utf-8").strip()
    preds, refs = [], []
    d_true, d_pred, s_true, s_pred = [], [], [], []
    detail = []
    for i, r in enumerate(rows, start=1):
        user_text = r.get("input", "").strip()
        if not user_text:
            user_text = build_user_text(r, normalize_bbox_payload(r.get("bbox_list", [])))
        messages = build_messages(
            resolve_data_path(r["visual_path"]),
            resolve_data_path(r["thermal_path"]),
            user_text,
            system_instruction=system_instruction,
        )
        inputs, prompt_len = qwen3vl_inputs_from_messages(processor, messages, model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                do_sample=False,
            )
        text = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        preds.append(text)
        refs.append(r["output"])
        gt_d = r.get("defect_type_gt", "Unknown")
        pd_d = pick_defect_label(text)
        gt_s = normalize_severity_gt(str(r.get("severity_gt", "")))
        pd_s = pick_severity(text)
        d_true.append(gt_d if gt_d in DEFECT_CLASSES else "Unspecified defect")
        d_pred.append(pd_d)
        s_true.append(gt_s if gt_s in SEV_CLASSES else "MINOR")
        s_pred.append(pd_s)
        detail.append(
            {
                "image_id": r["image_id"],
                "pred": text,
                "defect_true": gt_d,
                "defect_pred": pd_d,
                "severity_true": gt_s,
                "severity_pred": pd_s,
            }
        )
        if i % 20 == 0:
            print(f"evaluated={i}/{len(rows)}")

    bs_model = (args.bertscore_model or "").strip() or None
    out = {
        "num_eval": len(rows),
        "text_metrics": evaluate_texts(preds, refs, bertscore_model=bs_model),
        "defect_type_metrics": macro_prf1(d_true, d_pred, DEFECT_CLASSES),
        "severity_metrics": macro_prf1(s_true, s_pred, SEV_CLASSES),
        "detail": detail,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["num_eval", "text_metrics", "defect_type_metrics", "severity_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
