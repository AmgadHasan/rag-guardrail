from typing import List
import os
from openai import OpenAI
import polars as pl

DENYLIST_FILE = "denylist.txt"


def load_denylist(filepath: str) -> set[str]:
    """
    Load forbidden terms from a text file into a set for fast lookup.

    Each line in the file is stripped and converted to lowercase.
    """
    with open(filepath) as f:
        return {line.strip().lower() for line in f}


DENYLIST = load_denylist(DENYLIST_FILE)


def check_denylist(query: str) -> bool:
    """
    Check if the query contains any forbidden terms from the denylist.

    Performs a substring match on the lowercase query against each denylist term.
    """
    query_lower = query.lower()
    return any(forbidden in query_lower for forbidden in DENYLIST)


def embed_text(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using OpenAI's embedding model.

    Returns a list of embedding vectors, one for each input text.
    """
    model = os.getenv("OPENAI_EMBEDDING_MODEL")
    if not model:
        raise ValueError("OPENAI_EMBEDDING_MODEL environment variable must be set")
    response = client.embeddings.create(input=texts, model=model)
    return [emb.embedding for emb in response.data]


def create_corpus_dataframe(client: OpenAI, texts: List[str]) -> pl.DataFrame:
    """
    Create a Polars DataFrame representing the corpus with texts and their embeddings.
    """
    embeddings = embed_text(client, texts)
    if not embeddings:
        emb_dim = 0
    else:
        emb_dim = len(embeddings[0])
    schema = {"text": pl.Utf8, "embedding": pl.Array(pl.Float64, emb_dim)}
    return pl.DataFrame({"text": texts, "embedding": embeddings}, schema=schema)


# TODO: Implement similarity search and retrieval
def retrieve_snippets(
    query_embedding: List[float],
    corpus_embeddings: List[List[float]],
    corpus: List[str],
    k: int,
) -> List[str]:
    """
    Retrieve the top-k most similar snippets from the corpus based on cosine similarity.

    Returns a list of snippet texts.
    """
    pass


# TODO: Implement answer generation (naive concatenation for prototype)
def generate_answer(client: OpenAI, query: str, snippets: List[str]) -> str:
    """
    Generate an answer based on the query and retrieved snippets using OpenAI chat completion.

    Currently concatenates the snippets, but can be extended for more advanced QA.
    """
    # TODO: Implement actual OpenAI chat completion
    combined_text = "\n".join(snippets)
    return f"Based on the context:\n{combined_text}\n\nQuery: {query}"
