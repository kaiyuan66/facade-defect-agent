#!/usr/bin/env python3
"""LoRA fine-tuning for multimodal (RGB + thermal) facade data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from vl_multimodal_utils import build_messages, build_user_text, normalize_bbox_payload, resolve_data_path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SYSTEM_PROMPT = _REPO_ROOT / "RAG-knowledge-base/prompt/system_prompt.md"


@dataclass
class TrainConfig:
    model_path: str
    train_file: str
    output_dir: str
    max_length: int = 2048
    lr: float = 2e-5
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    gradient_checkpointing: bool = False
    system_prompt_file: str = str(_DEFAULT_SYSTEM_PROMPT)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="LoRA fine-tune template for facade diagnosis.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True, help="jsonl with instruction/input/output")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="开启梯度检查点，降低显存（大模型如 Qwen3-VL-8B 建议开启）",
    )
    p.add_argument(
        "--system_prompt_file",
        type=str,
        default=str(_DEFAULT_SYSTEM_PROMPT),
    )
    args = p.parse_args()
    return TrainConfig(
        model_path=args.model_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        system_prompt_file=args.system_prompt_file,
    )


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class Qwen3VLMultimodalCollator:
    def __init__(self, processor, max_length: int, system_instruction: str):
        self.processor = processor
        self.max_length = max_length
        self.system_instruction = system_instruction

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError("Use batch_size=1 for this multimodal collator.")
        ex = features[0]
        visual_path = resolve_data_path(ex["visual_path"])
        thermal_path = resolve_data_path(ex["thermal_path"])
        user_text = ex.get("input", "").strip()
        if not user_text:
            bbox_json = normalize_bbox_payload(ex.get("bbox_list", []))
            user_text = build_user_text(ex, bbox_json)
        messages = build_messages(visual_path, thermal_path, user_text, system_instruction=self.system_instruction)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["output"]}]})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        from PIL import Image

        rgb = Image.open(visual_path).convert("RGB")
        thermal = Image.open(thermal_path).convert("RGB")
        enc = self.processor(
            text=[text],
            images=[rgb, thermal],
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc


def main() -> None:
    cfg = parse_args()
    processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
    model_cfg = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
    mt = getattr(model_cfg, "model_type", "")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if mt == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    elif mt in ("deepseek_vl", "deepseek_vl_hybrid"):
        # HF 原生架构；勿用旧版 multi_modality 本地权重（AutoConfig 无法识别）
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    ds_rows = load_jsonl(Path(cfg.train_file).resolve())
    ds = Dataset.from_list(ds_rows)
    system_instruction = Path(cfg.system_prompt_file).read_text(encoding="utf-8").strip()
    collator = Qwen3VLMultimodalCollator(processor, cfg.max_length, system_instruction=system_instruction)

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        save_strategy="epoch",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
