from fastapi import FastAPI, HTTPException
from typing import List
from openai import OpenAI
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.schemas import QueryRequest, Snippet, AnswerResponse
from src.utils import contains_forbidden_terms, embed_text, CORPUS
from src.core import generate_response

app = FastAPI(title="RAG Guardrail API")

chat_client: OpenAI = None
embedding_client: OpenAI = None
qdrant_client: QdrantClient = None
collection_name: str = None


@app.on_event("startup")
def startup_event():
    """
    Initialize OpenAI clients and Qdrant client. Create collection if needed and embed corpus.
    """
    global chat_client, embedding_client, qdrant_client, collection_name
    # chat
    chat_api_key = os.getenv("OPENAI_CHAT_API_KEY")
    chat_base_url = os.getenv("OPENAI_CHAT_BASE_URL")
    # Embeddings
    embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")

    if not chat_api_key:
        raise ValueError("OPENAI_CHAT_API_KEY environment variable must be set")
    if not embedding_api_key:
        raise ValueError("OPENAI_EMBEDDING_API_KEY environment variable must be set")
    if not embedding_model:
        raise ValueError("OPENAI_EMBEDDING_MODEL environment variable must be set")

    chat_client = OpenAI(api_key=chat_api_key, base_url=chat_base_url)
    embedding_client = OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
    qdrant_client = QdrantClient(path="./qdrant_db")
    collection_name = embedding_model

    # Check if collection exists, if not, create and populate
    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating and populating collection '{collection_name}'.")
        embeddings = embed_text(embedding_client, CORPUS)
        vector_size = len(embeddings[0])

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=qmodels.Distance.COSINE
            ),
        )

        points = [
            qmodels.PointStruct(
                id=idx,
                vector=emb,
                payload={"text": txt},
            )
            for idx, (txt, emb) in enumerate(zip(CORPUS, embeddings))
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)


@app.post("/answer", response_model=AnswerResponse)
def answer_query(req: QueryRequest) -> AnswerResponse:
    """
    Handle the /answer endpoint: process query, check guardrail, retrieve snippets, generate answer.
    """
    user_query = req.query
    if contains_forbidden_terms(user_query):
        raise HTTPException(
            status_code=422,
            detail="Query contains forbidden terms. Please rephrase your question.",
        )

    global qdrant_client, collection_name
    top_snippets, generated_answer = generate_response(
        query=user_query,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        embedding_client=embedding_client,
        chat_client=chat_client,
    )

    return AnswerResponse(
        snippets=[Snippet(text=s.text, score=s.score) for s in top_snippets],
        answer=generated_answer,
    )
