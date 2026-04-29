# Interpretable Multimodal Agents for Facade Defect Detection

![Interpretable multimodal agents overview](asset/overview1.jpg)

## What Is the Interpretable Multimodal Agent?

An end-to-end interpretable multimodal agent framework that fuses **3D digital twins**, **visual–thermal perception**, and **RAG-driven engineering guidelines** for transparent building defect prognosis.

---

## Repository layout

| Path | Role |
|------|------|
| `agent/` | Training and evaluation scripts (`lora_finetune_vlm.py`, `evaluate_local_peft.py`, `evaluate_ft_models.py`, `run_vl_rag_batch.py`) and `vl_multimodal_utils.py`. |
| `agent/RAG-knowledge-base/` | Prompts, textbook PDFs, embedding code, and `embedding/store/` vector index. |
| `dataset/` | Mesh alignment JSON, multimodal jsonl, `image/` (sessions, `labels.jsonl`). |
| `result/` | Example metrics and LoRA checkpoints. |
| `asset/` | Figures (e.g. overview image). |
| `utils/` | Shared metrics. |

Run Python entrypoints from the **repository root**.

---

## Environment setup

```bash
cd /path/to/facade-defect-agent   # your clone
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

---

## Create your own knowledge base (RAG)

1. Place **PDF** textbooks under `agent/RAG-knowledge-base/textbook/` (or pass `--textbook_dir`).

2. **Build the vector store** (from repository root):

   ```bash
   export FACADE_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
   python agent/RAG-knowledge-base/embedding/build_vector_store.py \
     --textbook_dir agent/RAG-knowledge-base/textbook \
     --output_dir agent/RAG-knowledge-base/embedding/store \
     --model_path "$FACADE_EMBEDDING_MODEL"
   ```

3. The embedding model used at **build** time must match **retrieval** time (same vector dimension). Adjust `--chunk_size` / `--overlap` in `build_vector_store.py` if needed.

4. Edit `agent/RAG-knowledge-base/prompt/system_prompt.md` and `cot_prompt.md` for domain-specific instructions.

More detail: `agent/RAG-knowledge-base/README.md`.

---

## Multimodal inference

### Batch RGB + thermal + RAG (OpenAI-compatible server)

```bash
python agent/run_vl_rag_batch.py \
  --model Qwen3-VL-8B-Instruct \
  --label_json dataset/image/labels.jsonl \
  --image_root dataset/image/your_session \
  --store_dir agent/RAG-knowledge-base/embedding/store \
  --out_dir result/my_vl_rag_run \
  --num_locations 50
```

Defaults point to **`dataset/image/labels.jsonl`**, **`dataset/image/20251003094651`**, and **`agent/RAG-knowledge-base/embedding/store`**. Optional: set **`FACADE_EMBEDDING_MODEL`** for RAG embeddings.

### Local LoRA evaluation

```bash
python agent/evaluate_local_peft.py \
  --base_model Qwen/Qwen3-VL-2B-Instruct \
  --lora_path result/your_run/lora \
  --test_file dataset/text/your_test.jsonl \
  --out_file result/your_eval.json
```

### LoRA fine-tuning

```bash
python agent/lora_finetune_vlm.py \
  --model_path Qwen/Qwen3-VL-2B-Instruct \
  --train_file dataset/text/your_train.jsonl \
  --output_dir result/my_lora
```
