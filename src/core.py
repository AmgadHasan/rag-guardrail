from typing import List
from openai import OpenAI
from src.utils import embed_text, generate_chat_response
from src.schemas import Snippet
from textwrap import dedent


def retrieve_documents(
    query: str,
    k: int,
    qdrant_client,
    collection_name: str,
    embedding_client: OpenAI,
) -> List[Snippet]:  # ← return type updated
    """
    Retrieve top‑k relevant snippets from the corpus using Qdrant vector search.

    Returns:
        List of Snippet objects (text + score).
    """
    query_embedding = embed_text(embedding_client, [query])[0]

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k,
    )

    # Build Snippet objects with text and score from each hit
    return [
        Snippet(text=hit.payload["text"], score=hit.score) for hit in search_results
    ]


def generate_answer(query: str, snippets: List[Snippet], chat_client: OpenAI) -> str:
    """
    Generate an answer using OpenAI chat client based on query and retrieved snippets.

    Args:
        query: The user query.
        snippets: Retrieved context snippets.
        chat_client: OpenAI client for chat completions.

    Returns:
        Generated answer string.
    """
    context = "\n".join([s.text for s in snippets])
    user_message = dedent(
        f"""
        Please provide a helpful answer to the query based on the provided context.

        # Context:
        ```
        {context}
        ```

        # Query:
        ```
        {query}
        ```
        """
    )
    model_response = generate_chat_response(
        client=chat_client, messages=[{"role": "user", "content": user_message}]
    )
    return model_response


def generate_response(
    query: str,
    qdrant_client,
    collection_name: str,
    embedding_client: OpenAI,
    chat_client: OpenAI,
) -> (List[str], str):
    """
    Retrieve relevant snippets and generate an answer.

    Uses Qdrant search with k=3 by default.
    """
    top_snippets = retrieve_documents(
        query,
        k=3,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        embedding_client=embedding_client,
    )
    generated_answer = generate_answer(query, top_snippets, chat_client)
    return top_snippets, generated_answer
