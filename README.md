# NLP-RAG

NLP course project: **multimodal RAG** over **PDFs** (text, tables, images)—extract, chunk, enrich with a local LLM, index in Chroma, retrieve.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13 per `pyproject.toml`)
- [Ollama](https://ollama.com/) with:
  - **`nomic-embed-text`** — embeddings for Chroma (`rag/database.py`)
  - **`gemma3:4b`** — summaries over mixed text / tables / images (`rag/enhance.py`)

```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

GPU-oriented dependencies (`onnxruntime-gpu`, `tensorrt-cu12`, PyTorch CUDA) are configured for Linux/Windows; adjust `pyproject.toml` if you run CPU-only.

## Environment setup

### 1. Environment variables

Create a **`.env`** file in the project root (it is listed in `.gitignore`).

Add your Hugging Face token—required when pulling models from the Hub (e.g. Unstructured / layout pipelines):

```env
HF_TOKEN=hf_your_token_here
```

Create a token under your account: [Hugging Face — Access Tokens](https://huggingface.co/settings/tokens).

### 2. PDF documents directory

You need a **`docs/`** directory **locally**.

- Create `docs/`
- Copy the PDFs you want to run through the pipeline there (e.g. research papers)

Paths in code resolve `docs/` relative to the repository root.

## How to use

All commands below assume the **repository root** as the current working directory.

1. **Install dependencies**

```bash
uv sync
```

2. **Configure** `.env` with `HF_TOKEN` and ensure Ollama models are pulled (see above).

3. **Build the vector index** (PDFs → chunks → LLM-enhanced documents → Chroma under `db/`):

```bash
uv run python rag/database.py
```

Executing `rag/database.py` as a script runs `process_pdfs()` (see `rag/database.py`).

4. **Try retrieval** (loads `db/chroma_db_v1` and runs a sample query):

```bash
uv run python rag/main.py
```

The notebook `notebooks/01_process_pdfs.ipynb` remains the exploratory / step-by-step pipeline.

## Project layout

```
.
├── README.md
├── pyproject.toml          # project metadata and dependencies (uv)
├── uv.lock
├── rag/                    # application modules (run via paths below from repo root)
│   ├── load_data.py        # Unstructured PDF partition + chunk_by_title
│   ├── enhance.py          # multimodal chunk summaries via ChatOllama
│   ├── database.py         # Chroma + OllamaEmbeddings; build DB
│   └── main.py             # minimal retriever smoke test
├── docs/                   # source PDFs (local only; not in the repo)
├── db/                     # Chroma persist dir (local only; gitignored)
├── notebooks/
│   └── 01_process_pdfs.ipynb
└── .env                    # HF_TOKEN, etc. (local only)
```

After cloning, create **`docs/`**, **`.env`**, and run the build step to populate **`db/`**—those artifacts are not committed.
