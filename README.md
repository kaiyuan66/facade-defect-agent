# Interpretable Multimodal Agents for Facade Defect Detection in Digital Twins

[![Overview figure (PDF)](Figure/overview.pdf)](Figure/overview.pdf)

*Overview: open [`Figure/overview.pdf`](Figure/overview.pdf) locally for the full-resolution figure. GitHub’s Markdown preview may not render embedded PDFs as images.*

This repository supports **paired RGB + thermal** facade inspection, **3D / BIM-aligned metadata**, **explainable reporting** (severity, defect type, engineering narrative), optional **LoRA fine-tuning** of vision–language models, and a **textbook-style RAG** pipeline over PDF knowledge sources.

---

## Repository layout

| Path | Role |
|------|------|
| `Agent/` | Training (`lora_finetune_vlm.py`), local/API evaluation, batch **VLM + RAG** inference (`run_vl_rag_batch.py`), shared `vl_multimodal_utils.py`. |
| `Demo-data/` | Example mesh alignment JSON, detection labels, and image folder layout for demos. |
| `Figure/` | Paper / overview figures (`overview.pdf`, etc.). |
| `RAG-knowledge-base/` | Prompts, `textbook/` PDFs, embedding scripts, and `embedding/store/` vector index. |
| `Results/` | Example evaluation exports and LoRA checkpoints (large files; optional to keep). |
| `utils/` | Shared metrics (BLEU, ROUGE-L, BERTScore, macro P/R/F1). |

All Python entrypoints resolve paths **relative to the repository root** (or accept your own absolute paths). Optional environment variable:

- `FACADE_EMBEDDING_MODEL` — Hugging Face **model id** or **local directory** for `sentence-transformers` when building or querying the RAG index (default: `Qwen/Qwen3-Embedding-0.6B`).

---

## Environment setup

**pip**

```bash
cd /path/to/facade-agent   # your clone
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**conda**

```bash
conda env create -f environment.yml
conda activate facade-agent
```

Install a **CUDA-enabled** `torch` build that matches your GPU driver if you train or run local `evaluate_local_peft.py`.

For **serving** a VLM with vLLM (optional), install vLLM following the official instructions for your OS/CUDA version; then point scripts at `--base_url` (OpenAI-compatible `.../v1`).

---

## Bring your own image data

1. **Folder layout** (same convention as `Demo-data/image/<session_id>/`):

   - `<image_root>/visual/` — RGB (or visible) images, filenames ending with `_V.jpg`.
   - `<image_root>/thermal/` — aligned thermal images, same basename with `_T.jpg`.

   Subfolders under `visual/` and `thermal/` are allowed; the code discovers files by `rglob`.

2. **Detection / weak labels (optional but used by batch RAG script)**  
   A JSON array of objects like:

   ```json
   { "filename": "DJI_xxx_0001_V.jpg", "label": "Crack", "bbox": [x, y, w, h] }
   ```

   with **normalized xywh** in image coordinates (see `Demo-data/image/labels-export-2026-04-28.json`).

3. **Training / evaluation jsonl**  
   Each line should include at least `visual_path`, `thermal_path`, `instruction`, `input`, `output`, and identifiers as in `Demo-data/text/3Dmeshalign_multimodal_gt.jsonl`. Paths may be **relative to the repo root** (recommended) or absolute.

4. **3D / digital-twin alignment**  
   Tabular or JSON records (e.g. `Demo-data/text/3Dmeshalign.json`) with fields such as defect id, facade area, floor, coordinates, severity, type, and `Full_Image` linking to the RGB filename.

---

## Bring your own knowledge base (RAG)

1. Place **PDF** textbooks or standards under `RAG-knowledge-base/textbook/` (or pass `--textbook_dir`).

2. **Build the vector store** (from repository root):

   ```bash
   export FACADE_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"   # or a local path
   python RAG-knowledge-base/embedding/build_vector_store.py \
     --textbook_dir RAG-knowledge-base/textbook \
     --output_dir RAG-knowledge-base/embedding/store \
     --model_path "$FACADE_EMBEDDING_MODEL"
   ```

   This writes `embeddings.npy`, `metadata.jsonl`, and `store_config.json` under `--output_dir`.

3. **Chunking / model choice**  
   Default chunking is tuned for long PDF pages; adjust `--chunk_size` and `--overlap` in `build_vector_store.py` if needed. The embedding model used at **build** time must be the **same** as at **retrieval** time (dimension consistency).

4. **Prompts**  
   Edit `RAG-knowledge-base/prompt/system_prompt.md` and `cot_prompt.md` for domain-specific tone and safety constraints.

---

## Multimodal inference

### A. Batch RGB + thermal + RAG (OpenAI-compatible server)

Requires a running VLM (e.g. vLLM) and a built RAG store.

```bash
python Agent/run_vl_rag_batch.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-8B-Instruct \
  --label_json Demo-data/image/your-labels.json \
  --image_root Demo-data/image/your_session \
  --store_dir RAG-knowledge-base/embedding/store \
  --embedding_model "${FACADE_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}" \
  --out_dir Results/my_vl_rag_run \
  --num_locations 50
```

### B. Local Hugging Face + LoRA (no server)

```bash
python Agent/evaluate_local_peft.py \
  --base_model Qwen/Qwen3-VL-2B-Instruct \
  --lora_path Results/your_run/lora \
  --test_file Demo-data/text/your_test.jsonl \
  --out_file Results/your_eval.json
```

### C. Remote fine-tuned model via API

```bash
python Agent/evaluate_ft_models.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model your-served-model-name \
  --test_file Demo-data/text/your_test.jsonl \
  --out_file Results/your_api_eval.json
```

### D. LoRA fine-tuning

```bash
python Agent/lora_finetune_vlm.py \
  --model_path Qwen/Qwen3-VL-2B-Instruct \
  --train_file Demo-data/text/your_train.jsonl \
  --output_dir Results/my_lora \
  --epochs 1
```

Run these commands from the **repository root** so imports (`utils`, `vl_multimodal_utils`) and default relative paths resolve correctly.

---

## Generic RAG QA (text-only over the store)

```bash
cd RAG-knowledge-base
python rag_agent.py \
  --question "Your question referencing facade codes or maintenance." \
  --base_url http://127.0.0.1:8000/v1 \
  --llm_model Qwen3-VL-8B-Instruct
```

---

## License and citation

Add your license and preferred citation text here when publishing.

---

