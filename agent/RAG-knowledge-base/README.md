# RAG Knowledge Base

Paths below are relative to the **repository root** unless you pass absolute paths.

## 1) Build vector store

```bash
export FACADE_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
python agent/RAG-knowledge-base/embedding/build_vector_store.py \
  --textbook_dir agent/RAG-knowledge-base/textbook \
  --output_dir agent/RAG-knowledge-base/embedding/store \
  --model_path "$FACADE_EMBEDDING_MODEL"
```

Optional: set `FACADE_EMBEDDING_MODEL` instead of `--model_path` to choose a local or HuggingFace embedding model.

## 2) Query with generic RAG agent

```bash
cd agent/RAG-knowledge-base
python rag_agent.py \
  --question "A facade crack is located at 15m elevation with ΔT=1.4°C. How should risk and maintenance be assessed?" \
  --base_url http://127.0.0.1:8000/v1 \
  --llm_model Qwen3-VL-8B-Instruct
```

## 3) Prompt files

- `prompt/system_prompt.md`: system-level safety/compliance constraints.
- `prompt/cot_prompt.md`: CoT reasoning scaffold and output template.
