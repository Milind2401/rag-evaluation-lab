import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from src.config import (
    AZURE_AI_SEARCH_ENDPOINT,
    AZURE_AI_SEARCH_API_KEY,
    AZURE_AI_SEARCH_INDEX_NAME,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)
from src.embeddings.azure_openai import get_embedding_client


def _get_credential():
    return AzureKeyCredential(AZURE_AI_SEARCH_API_KEY)


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=AZURE_AI_SEARCH_ENDPOINT,
        index_name=AZURE_AI_SEARCH_INDEX_NAME,
        credential=_get_credential(),
    )


def get_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=AZURE_AI_SEARCH_ENDPOINT,
        credential=_get_credential(),
    )


def create_index(dimension: int = 1536):
    """Create or recreate the search index."""
    index_client = get_index_client()

    try:
        index_client.get_index(AZURE_AI_SEARCH_INDEX_NAME)
        index_client.delete_index(AZURE_AI_SEARCH_INDEX_NAME)
        print(f"  Deleted existing index: {AZURE_AI_SEARCH_INDEX_NAME}")
    except Exception:
        pass

    index = SearchIndex(
        name=AZURE_AI_SEARCH_INDEX_NAME,
        fields=[
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="text", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SearchField(name="section", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="strategy", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimension,
                vector_search_profile_name="default",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="default")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="default")],
        ),
    )

    index_client.create_index(index)
    print(f"  Created index: {AZURE_AI_SEARCH_INDEX_NAME}")


def index_chunks(chunks: list, embeddings: list[list[float]], strategy: str):
    """Index chunks with their embeddings into Azure AI Search."""
    import re
    search_client = get_search_client()

    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        # Sanitize ID: only allow letters, digits, underscore, dash, equals
        safe_id = re.sub(r'[^a-zA-Z0-9_=-]', '_', chunk.id)
        doc = {
            "id": safe_id,
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section,
            "strategy": strategy,
            "embedding": embedding,
        }
        documents.append(doc)

    # Upload in batches of 100
    for i in range(0, len(documents), 100):
        batch = documents[i:i + 100]
        search_client.upload_documents(batch)

    print(f"  Indexed {len(documents)} documents")


def vector_search(query: str, top_k: int = 5, strategy_filter: str = None,
                  deployment_name: str = None) -> list[dict]:
    """Search using pure vector similarity."""
    search_client = get_search_client()
    model = deployment_name or AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    q_emb = get_embedding_client().embeddings.create(
        model=model, input=[query]
    ).data[0].embedding

    vector_query = VectorizedQuery(
        vector=q_emb,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    filter_expr = f"strategy eq '{strategy_filter}'" if strategy_filter else None

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        top=top_k,
        filter=filter_expr,
    )

    return [{"id": r["id"], "text": r["text"], "score": r["@search.score"]} for r in results]


def bm25_search(query: str, top_k: int = 5, strategy_filter: str = None) -> list[dict]:
    """Search using BM25 keyword ranking."""
    search_client = get_search_client()

    filter_expr = f"strategy eq '{strategy_filter}'" if strategy_filter else None

    results = search_client.search(
        search_text=query,
        top=top_k,
        filter=filter_expr,
    )

    return [{"id": r["id"], "text": r["text"], "score": r["@search.score"]} for r in results]


def hybrid_search(query: str, top_k: int = 5, strategy_filter: str = None,
                  deployment_name: str = None) -> list[dict]:
    """Search using combined vector + BM25."""
    search_client = get_search_client()
    model = deployment_name or AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    q_emb = get_embedding_client().embeddings.create(
        model=model, input=[query]
    ).data[0].embedding

    vector_query = VectorizedQuery(
        vector=q_emb,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    filter_expr = f"strategy eq '{strategy_filter}'" if strategy_filter else None

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=top_k,
        filter=filter_expr,
    )

    return [{"id": r["id"], "text": r["text"], "score": r["@search.score"]} for r in results]


def delete_all_documents(strategy_filter: str = None):
    """Delete all documents from the index."""
    search_client = get_search_client()

    if strategy_filter:
        results = search_client.search("*", filter=f"strategy eq '{strategy_filter}'", select=["id"])
    else:
        results = search_client.search("*", select=["id"])

    doc_ids = [{"id": r["id"]} for r in results]
    if doc_ids:
        search_client.delete_documents(doc_ids)
        print(f"  Deleted {len(doc_ids)} documents")
