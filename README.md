# NLP-RAG

NLP course project: **multimodal RAG** over **PDFs** (text, tables, images)—extract, chunk, enrich with a local LLM, index in Chroma, retrieve.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13 per `pyproject.toml`)
- [tesseract-ocr](https://github.com/tesseract-ocr/tesseract)
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

4. **Try RAG model** (loads `db/chroma_db_v1`, initializes retriever and llm generator, then runs a CLI app that lets you ask questions):

```bash
uv run python rag/main.py
```

## Available Flags / Options

| Flag | Description | Default / Notes |
|------|------------|----------------|
| `--llm` | Specify which ollama LLM to use for generation (e.g., `gpt-4`, `gpt-3.5`). Note: to use a model you need to first pull it with ollama | gemma3:4b|
| `--hybrid-search` | Enables hybrid-search (semantic + keyword/dense + sparse) | offf |
| `--verbose` | All retrieved documents and metadata will be printed | Off |


The notebook `notebooks/01_process_pdfs.ipynb` remains the exploratory / step-by-step pipeline.

## RAG evaluation with RAGAS

### 1. Setup
To evaluate your RAG setup, you need to provide a test set, which contains sample queries and the expected reponses. The test set should be placed in a JSON file called `testset.json` in the folder `/testset`. The file should follow this format:
```json
[
  {
    "query": "What is the capital of Germany?",
    "expected_answer": "Berlin"
  },
  {
    "query": "Who wrote the novel '1984'?",
    "expected_answer": "George Orwell"
  },
  {
    "query": "When was the Eiffel Tower built?",
    "expected_answer": "1887–1889"
  }
]
```

### 2. Run evaluation
In command line run:
```bash
uv run python rag/evaluate.py
```
The script will return values of three key RAG metrics:
`context_recall`, `faithfulness`, `factual_correctness`




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
│   ├── retriever.py        # get retriever module
│   ├── generator.py        # get llm instance; generate reponse based on provided chunks
│   ├── evaluate.py         # run evaluation using RAGAS  
│   └── main.py             # CLI RAG runner
├── docs/                   # source PDFs (local only; not in the repo)
├── db/                     # Chroma persist dir (local only; gitignored)
├── notebooks/
│   └── 01_process_pdfs.ipynb
└── .env                    # HF_TOKEN, etc. (local only)
```

After cloning, create **`docs/`**, **`.env`**, and run the build step to populate **`db/`**—those artifacts are not committed.
