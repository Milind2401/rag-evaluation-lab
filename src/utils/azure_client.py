from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_AI_SEARCH_ENDPOINT,
    AZURE_AI_SEARCH_API_KEY,
    AZURE_AI_SEARCH_INDEX_NAME,
)


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=AZURE_AI_SEARCH_ENDPOINT,
        index_name=AZURE_AI_SEARCH_INDEX_NAME,
        credential=DefaultAzureCredential() if not AZURE_AI_SEARCH_API_KEY else AZURE_AI_SEARCH_API_KEY,
    )


def get_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=AZURE_AI_SEARCH_ENDPOINT,
        credential=DefaultAzureCredential() if not AZURE_AI_SEARCH_API_KEY else AZURE_AI_SEARCH_API_KEY,
    )
