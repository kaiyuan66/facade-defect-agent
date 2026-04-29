"""Evaluation metrics used in the paper (BLEU-4, ROUGE-L, BERTScore, P/R/F1)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def bleu4(pred: str, ref: str) -> float:
    """Sentence-level BLEU-4 with smoothing."""
    return float(
        sentence_bleu(
            [ref.split()],
            pred.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(ref, pred)["rougeL"].fmeasure
    return float(score)


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_bertscore_corpus(
    preds: Sequence[str],
    refs: Sequence[str],
    model_type: str = "Qwen/Qwen3-Embedding-0.6B",
    device: Optional[str] = None,
    batch_size: int = 8,
) -> dict:
    """
    语料级 BERTScore（默认用 Qwen3 官方 Embedding 0.6B 抽上下文向量）。
    需安装: pip install bert-score
    """
    if len(preds) != len(refs):
        raise ValueError("preds and refs length mismatch")
    try:
        import torch
        from bert_score import score as bert_score_fn
        from bert_score.utils import model2layers
        from transformers import AutoConfig
    except ImportError:
        return {"bertscore_f1": None, "bertscore_error": "bert-score 未安装"}

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = None
    if model_type not in model2layers:
        try:
            cfg = AutoConfig.from_pretrained(model_type, trust_remote_code=True)
            num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0) or None
        except Exception:
            num_layers = None
        if num_layers is None:
            num_layers = 24

    kwargs = dict(
        model_type=model_type,
        verbose=False,
        device=dev,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )
    if num_layers is not None:
        kwargs["num_layers"] = num_layers

    P, R, F1 = bert_score_fn(list(preds), list(refs), **kwargs)
    def _m(x):
        if hasattr(x, "mean"):
            t = x.mean()
            return float(t.cpu().item() if hasattr(t, "cpu") else float(t))
        return float(x)

    return {
        "bertscore_precision": _m(P),
        "bertscore_recall": _m(R),
        "bertscore_f1": _m(F1),
        "bertscore_model": model_type,
    }


def evaluate_texts(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    bertscore_model: Optional[str] = None,
    bertscore_device: Optional[str] = None,
    bertscore_batch_size: int = 8,
) -> dict:
    if len(preds) != len(refs):
        raise ValueError("preds and refs length mismatch")
    bleu_scores = [bleu4(p, r) for p, r in zip(preds, refs)]
    rouge_scores = [rouge_l(p, r) for p, r in zip(preds, refs)]
    n = len(preds) or 1
    out: dict = {
        "bleu4": sum(bleu_scores) / n,
        "rougeL": sum(rouge_scores) / n,
    }
    if bertscore_model:
        out.update(
            compute_bertscore_corpus(
                preds,
                refs,
                model_type=bertscore_model,
                device=bertscore_device,
                batch_size=bertscore_batch_size,
            )
        )
    return out


def classification_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    true_list: List[int] = list(y_true)
    pred_list: List[int] = list(y_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred length mismatch")
    tp = fp = fn = 0
    for t, p in zip(true_list, pred_list):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn}
