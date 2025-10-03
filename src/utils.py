from typing import List
import os
from openai import OpenAI

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


def generate_chat_response(client: OpenAI, messages: List[dict]) -> str:
    """
    Sends a list of chat messages to the OpenAI API and returns the assistant's reply.

    Parameters
    ----------
    client : OpenAI
        An instantiated OpenAI client (e.g., `OpenAI(api_key=...)`).
    messages : List[dict]
        A list of message dictionaries in the format expected by the API,
        e.g. [{"role": "user", "content": "Hello"}].

    Returns
    -------
    str
        The content of the assistant's response. Returns an empty string if the
        request fails.
    """
    model = os.getenv("OPENAI_CHAT_MODEL")
    try:
        response = client.chat.completions.create(
            model=model,  # or any model set in the client defaults
            messages=messages,
            temperature=0.1,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
