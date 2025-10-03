from typing import List
import os
from openai import OpenAI
import json

DENYLIST_FILE = "denylist.txt"
CORPUS_FILE = "corpus.json"


def load_corpus(filepath: str) -> List[str]:
    """
    Load the text corpus from a JSON file.

    The JSON file should contain a list of strings, e.g.
    ["sentence 1", "sentence 2", ...].
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_denylist(filepath: str) -> set[str]:
    """
    Load forbidden terms from a text file into a set for fast lookup.

    Each line in the file is stripped and converted to lowercase.
    """
    with open(filepath) as f:
        return {line.strip().lower() for line in f}


CORPUS = load_corpus(CORPUS_FILE)

DENYLIST = load_denylist(DENYLIST_FILE)


def contains_forbidden_terms(query: str) -> bool:
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
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
