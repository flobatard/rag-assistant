# RAG Assistant — Context for Claude

## Project overview

Local RAG (Retrieval-Augmented Generation) pipeline that indexes PDF documents, retrieves relevant chunks via semantic search, and answers questions using a local LLM served by Ollama.

## Architecture

```
PDFs (data/) → load_pdfs() → chunk_text() → SentenceTransformer embeddings → FAISS index
                                                                                    ↓
User query → encode → FAISS search → top-k chunks → Ollama LLM → answer
```

## Key files

| File | Role |
|------|------|
| `rag_pipeline.py` | Core pipeline: PDF loading, chunking (500 chars), embedding (`all-MiniLM-L6-v2`), FAISS indexing and search |
| `app.py` | Interactive CLI loop: builds index then accepts queries, prints raw retrieved chunks |
| `ask_local_llm.py` | LLM integration: sends context + question to local Ollama (`llama3.1:8b` via `localhost:11434`) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place PDF files in a `data/` folder (create it if absent).

## Running

```bash
# Search only (no LLM)
python app.py

# Full RAG with LLM (requires Ollama running)
# ollama serve
# ollama pull llama3.1:8b
python ask_local_llm.py
```

## Dependencies

- **faiss-cpu** — vector similarity index
- **sentence-transformers** — local embedding model (`all-MiniLM-L6-v2`)
- **pypdf** — PDF text extraction
- **requests** — HTTP calls to Ollama API
- **langchain / langchain-openai / openai** — present in requirements but not yet used in current code

## Notes for development

- The `data/` folder is not committed (add to `.gitignore` if needed).
- The index is rebuilt from scratch on every run — no persistence to disk yet.
- `ask_local_llm.py` is not yet wired to `app.py`; the two modules are independent.
- Chunk size is hardcoded to 500 characters in `rag_pipeline.py:9` — consider overlap for better retrieval.
- The `.env` file exists but is currently unused by the code (no `python-dotenv` call).
