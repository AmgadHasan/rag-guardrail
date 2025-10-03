from typing import List
import os
from openai import OpenAI
import polars as pl

DENYLIST_FILE = "denylist.txt"


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


def create_corpus_dataframe(client: OpenAI, corpus: List[str]) -> pl.DataFrame:
    """
    Embed the corpus texts and create a Polars DataFrame with text and embedding columns.
    """
    embeddings = embed_text(client, corpus)
    return pl.DataFrame({"text": corpus, "embedding": embeddings})
