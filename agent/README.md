# `agent/`

Python entrypoints for multimodal training, evaluation, and VL+RAG batch inference.

- **`vl_multimodal_utils.py`** — Messages, bbox formatting, `resolve_data_path()` (paths in jsonl are relative to the **repository root**).
- **`lora_finetune_vlm.py`** — LoRA fine-tuning.
- **`evaluate_local_peft.py`** — Local HF + LoRA evaluation.
- **`evaluate_ft_models.py`** — Evaluation via OpenAI-compatible HTTP API.
- **`run_vl_rag_batch.py`** — Batch RGB+thermal+RAG; defaults use `dataset/` and `agent/RAG-knowledge-base/`.

See the **[root README](../README.md)** for commands and directory layout.

Run scripts from the repository root, for example:

`python agent/run_vl_rag_batch.py --help`
