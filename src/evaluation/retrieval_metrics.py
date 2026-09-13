import numpy as np
from src.chunking.base import Chunk


def is_relevant(retrieved_chunk: Chunk, golden_entry: dict) -> bool:
    """Check if retrieved chunk contains the golden supporting text."""
    supporting_text = golden_entry.get("supporting_text", "")
    if not supporting_text:
        return False
    return supporting_text.lower() in retrieved_chunk.text.lower()


def recall_at_k(retrieved_chunks: list[Chunk], golden_entry: dict, k: int) -> float:
    """1.0 if any of top-k chunks is relevant, else 0.0."""
    for chunk in retrieved_chunks[:k]:
        if is_relevant(chunk, golden_entry):
            return 1.0
    return 0.0


def precision_at_k(retrieved_chunks: list[Chunk], golden_entry: dict, k: int) -> float:
    """Fraction of top-k chunks that are relevant."""
    relevant = sum(1 for chunk in retrieved_chunks[:k] if is_relevant(chunk, golden_entry))
    return relevant / k if k > 0 else 0.0


def mrr(retrieved_chunks: list[Chunk], golden_entry: dict) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant chunk."""
    for i, chunk in enumerate(retrieved_chunks):
        if is_relevant(chunk, golden_entry):
            return 1.0 / (i + 1)
    return 0.0


def compute_retrieval_metrics(
    all_results: list[list[Chunk]],
    golden_dataset: list[dict],
    k_values: list[int] = [1, 3, 5],
) -> dict:
    """Compute aggregated retrieval metrics across all questions."""
    metrics = {}
    for k in k_values:
        recalls = [recall_at_k(res, golden, k) for res, golden in zip(all_results, golden_dataset)]
        precisions = [precision_at_k(res, golden, k) for res, golden in zip(all_results, golden_dataset)]
        metrics[f"recall@{k}"] = np.mean(recalls)
        metrics[f"precision@{k}"] = np.mean(precisions)

    mrrs = [mrr(res, golden) for res, golden in zip(all_results, golden_dataset)]
    metrics["mrr"] = np.mean(mrrs)
    metrics["total_questions"] = len(golden_dataset)
    return metrics
