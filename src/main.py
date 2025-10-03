from fastapi import FastAPI, HTTPException
from typing import List
from openai import OpenAI
import os
import polars as pl

from src.schemas import QueryRequest, Snippet, AnswerResponse
from src.utils import check_denylist, create_corpus_dataframe, CORPUS
from src.core import generate_response

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
    user_query = req.query
    if check_denylist(user_query):
        raise HTTPException(
            status_code=422,
            detail="Query contains forbidden terms. Please rephrase your question.",
        )

    top_snippets, generated_answer = generate_response(
        query=user_query,
        corpus_df=corpus_df,
        embedding_client=embedding_client,
        chat_client=chat_client,
    )

    return AnswerResponse(
        snippets=[Snippet(text=s) for s in top_snippets], answer=generated_answer
    )
