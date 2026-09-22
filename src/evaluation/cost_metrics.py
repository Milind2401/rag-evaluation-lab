import math
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class LatencyTracker:
    """Track timing for each phase."""
    chunking_ms: float = 0.0
    embedding_generation_ms: float = 0.0
    indexing_ms: float = 0.0
    retrieval_ms: float = 0.0
    evaluation_ms: float = 0.0
    total_ms: float = 0.0

    def summary(self) -> dict:
        return {
            "chunking_ms": round(self.chunking_ms, 1),
            "embedding_generation_ms": round(self.embedding_generation_ms, 1),
            "indexing_ms": round(self.indexing_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "evaluation_ms": round(self.evaluation_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


# Mirrors the default batching in get_embeddings() (src/embeddings/azure_openai.py):
# one HTTP call per up-to-100 texts.
EMBEDDING_API_BATCH_SIZE = 100


@dataclass
class CostTracker:
    """Track API usage and estimated costs."""
    embedding_api_calls: int = 0
    embedding_tokens_input: int = 0
    embedding_cost_usd: float = 0.0
    cost_per_million: float = 0.02

    def add_embedding_batch(
        self,
        token_count: int,
        batch_size: int,
        api_calls: int = None,
    ):
        """
        Record an embedding usage event.

        batch_size: number of texts embedded in this event.
        api_calls: number of HTTP calls actually made. Defaults to
            ceil(batch_size / EMBEDDING_API_BATCH_SIZE), which matches
            get_embeddings()' batching (one call per up-to-100 texts).
            Pass an explicit value when the caller embeds differently,
            e.g. one API call per query -> api_calls=batch_size.
        """
        if api_calls is None:
            api_calls = math.ceil(batch_size / EMBEDDING_API_BATCH_SIZE) if batch_size > 0 else 0
        self.embedding_api_calls += api_calls
        self.embedding_tokens_input += token_count
        self.embedding_cost_usd += (token_count / 1_000_000) * self.cost_per_million

    def summary(self) -> dict:
        return {
            "embedding_api_calls": self.embedding_api_calls,
            "embedding_tokens_input": self.embedding_tokens_input,
            "embedding_cost_usd": round(self.embedding_cost_usd, 6),
        }


@dataclass
class ExperimentResult:
    """Full experiment result with metrics, cost, and latency."""
    strategy: str
    config: dict
    chunk_count: int
    avg_chunk_tokens: float
    retrieval_metrics: dict
    cost: CostTracker = field(default_factory=CostTracker)
    latency: LatencyTracker = field(default_factory=LatencyTracker)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "config": self.config,
            "chunk_count": self.chunk_count,
            "avg_chunk_tokens": round(self.avg_chunk_tokens, 1),
            "retrieval_metrics": {k: round(v, 4) for k, v in self.retrieval_metrics.items()},
            "cost": self.cost.summary(),
            "latency": self.latency.summary(),
        }

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.strategy}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Saved: {path}")


def compute_index_stats(chunks: list) -> dict:
    """Compute basic statistics about a set of chunks."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(c.text)) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "avg_chunk_tokens": np.mean(token_counts) if token_counts else 0,
        "std_chunk_tokens": np.std(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
    }
