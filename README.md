# NLP-RAG

NLP course project: **multimodal RAG** over **PDFs** (text, tables, images)—extract, index, retrieve, answer.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13 per `pyproject.toml`)
- [Ollama](https://ollama.com/) — required; pull the model used in the notebook (e.g. `ollama pull gemma3:4b`)

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

The notebook paths are set to resolve files under `docs/`.

## How to use (workflow)

1. Install dependencies - run:

```bash
uv sync
```
2. Configure `.env` with `HF_TOKEN`
3. Create `docs/` and place your PDFs there


## Project layout

Structure inspired by **Cookiecutter Data Science**, simplified for this repo:

```
.
├── README.md
├── pyproject.toml          # project metadata and dependencies (uv)
├── uv.lock                 # locked dependency versions
├── main.py                 # small Unstructured sample (using `layout-parser-paper.pdf` from [here](https://github.com/Unstructured-IO/unstructured/tree/main/example-docs/pdf))
├── docs/                   # source PDFs (local only; not in the repo)
├── notebooks/
│   ├── 01_process_pdfs.ipynb   # main pipeline: PDF → chunks → embeddings → RAG
│   └── dbv1/               
└── .env                    # HF_TOKEN, etc. (local only)
```

After cloning, you must create **`docs/`**, **`.env`**, and any data artifacts yourself—they are not part of the repository.
