#!/usr/bin/env python3
"""Evaluate fine-tuned models on test split and output metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.metrics import evaluate_texts


DEFECT_CLASSES = [
    "Crack",
    "Spalling",
    "Sign of Degradation",
    "Glass Anomaly (Spot)",
    "Glass Anomaly (Line)",
    "Debonding",
    "Peeling Paint",
    "Incorrect Installation",
    "Unknown",
]
SEV_CLASSES = ["LOW", "MINOR", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--base_url", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_file", type=Path, required=True)
    p.add_argument("--max_samples", type=int, default=400)
    return p.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    arr = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            arr.append(json.loads(line))
    return arr


def pick_defect_label(text: str) -> str:
    for c in DEFECT_CLASSES:
        if c.lower() in text.lower():
            return c
    return "Unknown"


def pick_severity(text: str) -> str:
    m = re.search(r"(LOW|MINOR|MODERATE|HIGH|CRITICAL)", text.upper())
    return m.group(1) if m else "UNKNOWN"


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
    rows = load_jsonl(args.test_file)[: args.max_samples]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    preds = []
    refs = []
    defect_true, defect_pred = [], []
    sev_true, sev_pred = [], []
    detail = []
    for r in rows:
        prompt = f"{r['instruction']}\n\n{r['input']}"
        resp = client.chat.completions.create(
            model=args.model,
            temperature=0.2,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        pred = resp.choices[0].message.content or ""
        ref = r["output"]
        preds.append(pred)
        refs.append(ref)
        d_true = r.get("defect_type_gt", "Unknown")
        d_pred = pick_defect_label(pred)
        s_true = pick_severity(ref)
        s_pred = pick_severity(pred)
        defect_true.append(d_true if d_true in DEFECT_CLASSES else "Unknown")
        defect_pred.append(d_pred)
        sev_true.append(s_true if s_true in SEV_CLASSES else "UNKNOWN")
        sev_pred.append(s_pred)
        detail.append(
            {
                "image_id": r["image_id"],
                "defect_true": d_true,
                "defect_pred": d_pred,
                "severity_true": s_true,
                "severity_pred": s_pred,
                "pred_text": pred,
            }
        )

    text_metrics = evaluate_texts(preds, refs)
    defect_metrics = macro_prf1(defect_true, defect_pred, DEFECT_CLASSES)
    severity_metrics = macro_prf1(sev_true, sev_pred, SEV_CLASSES)
    out = {
        "model": args.model,
        "num_eval": len(rows),
        "text_metrics": text_metrics,
        "defect_type_metrics": defect_metrics,
        "severity_metrics": severity_metrics,
        "detail": detail,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["model", "num_eval", "text_metrics", "defect_type_metrics", "severity_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
