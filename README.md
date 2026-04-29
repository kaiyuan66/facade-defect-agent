# Interpretable Multimodal Agents for Facade Defect Detection in Digital Twins

![Interpretable multimodal agents overview](Figure/overview1.jpg)

This repository supports **paired RGB + thermal** facade inspection, **3D / BIM-aligned metadata**, **explainable reporting** (severity, defect type, engineering narrative), optional **LoRA fine-tuning** of vision–language models, and a **textbook-style RAG** pipeline over PDF knowledge sources.


## Environment setup

**pip**

```bash
cd /path/to/facade-agent   # your clone
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```


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
   
3. **Chunking / model choice**  
   Default chunking is tuned for long PDF pages; adjust `--chunk_size` and `--overlap` in `build_vector_store.py` if needed. The embedding model used at **build** time must be the **same** as at **retrieval** time (dimension consistency).

4. **Prompts**  
   Edit `RAG-knowledge-base/prompt/system_prompt.md` and `cot_prompt.md` for domain-specific tone and safety constraints.

---

## Multimodal inference

### Batch RGB + thermal + RAG (OpenAI-compatible server)

Requires a running VLM and a built RAG store.

```bash
python Agent/run_vl_rag_batch.py \
  --model Qwen3-VL-8B-Instruct \
  --label_json Demo-data/image/labels.jsonl \
  --image_root Demo-data/image/your_session \
  --store_dir RAG-knowledge-base/embedding/store \
  --out_dir Results/my_vl_rag_run \
  --num_locations 50
```

Annotations default to **`Demo-data/image/labels.jsonl`** (one JSON object per line: `filename`, `label`, `bbox`). Legacy **JSON array** files (`.json`) are still accepted.
