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

## Guardrail

We've implemented a denylist as a form of a guardaril mechanism. The list is implemented as a txt file, where each line contains a prohibited term. We perform a substring match on the lowercase query against each denylist term.

This is quite practical as it's a very simple implementation, yet it's quite practical to prevent undesired terms (slurs, political terms, explicit terms, etc).

However, it can be bypassed (using zero-width spaces, or translated versions of the text, etc) - a more sophisticated implementation is needed for a production scenario.

## Monitoring metrics

### 1. Latency

We can measure the latency of the endpoint (and the inner steps). We can use `time.perf_counter()` to get the start and end time of each function/step, and then measure the time elapsed. These can be tracked during logging and exported to a prometheus+graphana stack. Alternatively, we can use a framework like highligths/sentry to trace the endpoint.

For the LLM calls themselves, we can also monitor and trace them separetely using an OTel compliant framework or something like langfuse.

```
Client Request → /answer Endpoint → Response
                     ↓
         total_endpoint_latency
                     ↓
    ┌─────────────────────────────────┐
    │ 1. request_parsing + validation │  → input_latency
    │    (receive QueryRequest)       │
    └─────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────┐
    │ 2. Guardrail Check              │  → guardrail_latency
    │    (contains_forbidden_terms)   │
    └─────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────┐
    │ 3. Query Embedding              │  → embedding_latency
    │    (embed_text via OpenAI)      │
    └─────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────┐
    │ 4. Vector Search                │  → search_latency
    │    (qdrant_client.search)       │
    └─────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────┐
    │ 5. Answer Generation            │  → generation_latency
    │    (generate_chat_response via OpenAI) │
    └─────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────┐
    │ 6. Response Formatting          │  → response_latency
    │    (build AnswerResponse)       │
    └─────────────────────────────────┘
```

Key latency measurements for the `/answer` endpoint:
- **input_latency**: Time to parse and validate request
- **guardrail_latency**: Time for forbidden term check
- **embedding_latency**: Time for OpenAI embedding API call
- **search_latency**: Time for Qdrant vector search
- **generation_latency**: Time for OpenAI chat completion
- **response_latency**: Time to format JSON response

### 2. Recall@k

We can measure the recall@k of the retrieval service. We'll collect user queries and the retrieved snippets, and also generate ground truths for the retrieved snippets (which snippets the service is supposed to retrieve). We then calculate how many of thr ground truth snippets are present in the retrieved snippets:

```
recall@k = num_golden_snippets_found_in_retrieved_context / total_number_of_golden_snippets
```
This score is a float that ranges from 0 (no golden snippets retrieved) to 1 (all golden snippets retrieve)

This requires annotation to generate the labels for the collected queries. We can use human annotators, or use a capable LLM that is fed all the snippets in the corpus and asked to identify the golden snippets. This can be iterated upon to improve the LLM's accuracy.