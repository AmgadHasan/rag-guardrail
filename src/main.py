from fastapi import FastAPI, HTTPException
from typing import List
from openai import OpenAI
import os
import polars as pl

from src.schemas import QueryRequest, Snippet, AnswerResponse
from src.utils import check_denylist, create_corpus_dataframe, embed_text


CORPUS = [
    "Python is a versatile programming language known for its simplicity and readability.",
    "Machine learning involves training algorithms to identify patterns in data.",
    "FastAPI is a web framework for building APIs with Python.",
    "Embeddings convert text into numerical vectors for semantic understanding.",
    "Retrieval-augmented generation combines search with language model generation.",
    "Large Language Models (LLMs) are neural networks trained on massive text datasets.",
    "LLMs can generate human-like text based on prompts and context.",
    "The Transformer architecture uses self-attention mechanisms to process sequences.",
    "Transformers revolutionized natural language processing with parallel processing capabilities.",
    "BERT and GPT are both Transformer-based models with different architectures.",
    "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
    "Backpropagation is the core algorithm for training deep neural networks.",
    "Convolutional Neural Networks (CNNs) excel at processing grid-like data such as images.",
    "Fine-tuning adapts pre-trained LLMs for specific tasks or domains.",
    "Multi-head attention allows Transformers to focus on different parts of input simultaneously.",
]

corpus_df: pl.DataFrame = None

app = FastAPI(title="RAG Guardrail API")

chat_client: OpenAI = None
embedding_client: OpenAI = None


@app.on_event("startup")
def startup_event():
    """
    Initialize OpenAI clients and compute corpus embeddings into DataFrame on startup.
    """
    global chat_client, embedding_client, corpus_df
    chat_api_key = os.getenv("OPENAI_CHAT_API_KEY")
    chat_base_url = os.getenv("OPENAI_CHAT_BASE_URL")
    embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")

    if not chat_api_key:
        raise ValueError("OPENAI_CHAT_API_KEY environment variable must be set")
    if not embedding_api_key:
        raise ValueError("OPENAI_EMBEDDING_API_KEY environment variable must be set")

    chat_client = OpenAI(api_key=chat_api_key, base_url=chat_base_url)
    embedding_client = OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
    corpus_df = create_corpus_dataframe(embedding_client, CORPUS)


@app.post("/answer", response_model=AnswerResponse)
def answer_query(req: QueryRequest) -> AnswerResponse:
    """
    Handle the /answer endpoint: process query, check guardrail, retrieve snippets, generate answer.
    """
    if check_denylist(req.query):
        raise HTTPException(
            status_code=422,
            detail="Query contains forbidden terms. Please rephrase your question.",
        )

    # TODO: Uncomment and implement retrieval and generation
    # query_emb = embed_text(embedding_client, [req.query])[0]
    # top_snippets = retrieve_snippets(query_emb, CORPUS_EMBEDDINGS, CORPUS, k=3)
    # gen_answer = generate_answer(chat_client, req.query, top_snippets)

    # Placeholder for now - return fixed response until RAG is implemented
    top_snippets = [
        "Placeholder snippet 1",
        "Placeholder snippet 2",
        "Placeholder snippet 3",
    ]
    gen_answer = "Placeholder naive answer"

    return AnswerResponse(
        snippets=[Snippet(text=s) for s in top_snippets], answer=gen_answer
    )
