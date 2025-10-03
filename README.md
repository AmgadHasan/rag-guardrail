# rag-guardrail

An AI‑powered question answering API that retrieves context with vector search and generates answers via an LLM. A guardrail blocks queries containing forbidden terms. Built using FastAPI

## Features
- **Vector search** over a text corpus for relevant snippets.
- **LLM** generation of concise, helpful answers.
- **Guardrail** filtering of disallowed terms.
- **FastAPI** endpoints with automatic Swagger UI.

## Prerequisites
- Python 3.10+
- **uv** (recommended) or any compatible Python runner.

## Setup
1. Copy the example environment file:
```sh
cp .env.example .env
```
2. Fill in the required variables in `.env` (OpenAI key, vector DB config, etc.).

## Running the Service
```sh
bash run.sh
```
or directly with uv:
```sh
uv run --env-file .env uvicorn src.main:app
```

The API will be available at `http://127.0.0.1:8000`. Access the interactive documentation at `http://127.0.0.1:8000/docs`.

## Usage
- **POST /answer** – Submit a question and get answers and relevant chunks.
- The service returns the answer and the source snippets used.

## License
MIT License. See `LICENSE` for details.