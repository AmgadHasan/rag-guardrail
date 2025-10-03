# rag-guardrail

This project provides an AI-powered question answering API. It uses vector search to find relevant information from a text corpus. OpenAI generates helpful answers based on the retrieved content. A guardrail prevents queries containing forbidden terms. Built with FastAPI for simple integration.

# How to run
This project requires `uv`


1. Copy the env file:
```sh
cp .env.example .env
```
2. Enter the needed env variables

3. Run the code:
```sh
bash run.sh
```
or
```
uv run --env-file .env uvicorn src.main:app
```

Then head to http://127.0.0.1:8000/docs to view the swagger docs page
