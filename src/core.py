from typing import List, Literal
import os
import numpy as np
from openai import OpenAI
import polars as pl
from src.utils import embed_text


def retrieve_documents(
    query: str,
    k: int,
    similarity_method: Literal["dot", "cosine", "l2"],
    corpus_df: pl.DataFrame,
    embedding_client: OpenAI,
) -> List[str]:
    """
    Retrieve top-k relevant snippets from the corpus based on query similarity.

    Args:
        query: The query text to search for.
        k: Number of top snippets to retrieve.
        similarity_method: Similarity method, one of "dot", "cosine", "l2".
        corpus_df: Polars DataFrame containing 'text' and 'embedding' columns.
        embedding_client: OpenAI client for embedding.

    Returns:
        List of top-k snippet texts.
    """
    query_embedding = embed_text(embedding_client, [query])[0]

    # Compute similarity for each embedding
    def compute_similarity(emb: List[float]) -> float:
        if similarity_method == "dot":
            return np.dot(query_embedding, emb)
        elif similarity_method == "cosine":
            norm_query = np.linalg.norm(query_embedding)
            norm_emb = np.linalg.norm(emb)
            return (
                np.dot(query_embedding, emb) / (norm_query * norm_emb)
                if norm_emb > 0
                else 0
            )
        elif similarity_method == "l2":
            return -np.linalg.norm(np.array(query_embedding) - np.array(emb))
        else:
            raise ValueError(f"Invalid similarity method: {similarity_method}")

    similarities = corpus_df["embedding"].map_elements(
        compute_similarity, return_dtype=pl.Float64
    )

    # Add similarities and sort
    temp_df = corpus_df.with_columns(similarities.alias("sim"))
    top_df = temp_df.sort("sim", descending=True).head(k)
    return top_df["text"].to_list()


def generate_answer(query: str, snippets: List[str], chat_client: OpenAI) -> str:
    """
    Generate an answer using OpenAI chat client based on query and retrieved snippets.

    Args:
        query: The user query.
        snippets: Retrieved context snippets.
        chat_client: OpenAI client for chat completions.

    Returns:
        Generated answer string.
    """
    context = "\n".join(snippets)
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nPlease provide a helpful answer based on the context."
    model = os.getenv("OPENAI_CHAT_MODEL")
    if not model:
        raise ValueError("OPENAI_CHAT_MODEL environment variable must be set")
    response = chat_client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def generate_response(
    query: str, corpus_df: pl.DataFrame, embedding_client: OpenAI, chat_client: OpenAI
) -> (List[str], str):
    """
    Retrieve relevant snippets and generate an answer.

    Uses cosine similarity with k=5 by default.
    """
    top_snippets = retrieve_documents(
        query,
        k=5,
        similarity_method="cosine",
        corpus_df=corpus_df,
        embedding_client=embedding_client,
    )
    generated_answer = generate_answer(query, top_snippets, chat_client)
    return top_snippets, generated_answer
