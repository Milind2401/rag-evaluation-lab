import numpy as np
from openai import AzureOpenAI
from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)


def get_embedding_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def get_embeddings(texts: list[str], batch_size: int = 100, deployment: str = None) -> list[list[float]]:
    """Get embeddings for a list of texts using Azure OpenAI."""
    client = get_embedding_client()
    model = deployment or AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def get_query_embedding(text: str, deployment: str = None) -> list[float]:
    """Get embedding for a single query text."""
    client = get_embedding_client()
    model = deployment or AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    response = client.embeddings.create(
        model=model,
        input=[text],
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
