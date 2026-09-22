import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    GOLDEN_DATASET_PATH,
    SOURCE_DOCS_DIR,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_LARGE_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_ADA_DEPLOYMENT,
)

EMBEDDING_MODELS = {
    "text-embedding-3-small": {
        "deployment": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "dimension": 1536,
        "cost_per_million_tokens": 0.02,
        "max_tokens": 8191,
        "description": "Budget - baseline from Phase 1",
    },
    "text-embedding-3-large": {
        "deployment": AZURE_OPENAI_EMBEDDING_LARGE_DEPLOYMENT,
        "dimension": 3072,
        "cost_per_million_tokens": 0.13,
        "max_tokens": 8191,
        "description": "Premium - highest quality",
    },
    "text-embedding-ada-002": {
        "deployment": AZURE_OPENAI_EMBEDDING_ADA_DEPLOYMENT,
        "dimension": 1536,
        "cost_per_million_tokens": 0.10,
        "max_tokens": 8191,
        "description": "Legacy - previous generation",
    },
}

STRATEGY_PREFIX = "phase2"
RETRIEVAL_TOP_K = 10
K_VALUES = [1, 3, 5, 10]
SEMANTIC_THRESHOLD = 0.6
EMBEDDING_BATCH_SIZE = 100

RESULTS_DIR = Path(__file__).parent / "results"
