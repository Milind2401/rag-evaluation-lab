import numpy as np

from .ndcg import ndcg_at_k


def is_relevant(retrieved_text: str, golden_entry: dict) -> bool:
    """Check if retrieved chunk contains the golden supporting text."""
    supporting_text = golden_entry.get("supporting_text", "")
    if not supporting_text:
        return False
    return supporting_text.lower() in retrieved_text.lower()


def recall_at_k(retrieved_texts: list[str], golden_entry: dict, k: int) -> float:
    """1.0 if any of top-k chunks is relevant, else 0.0."""
    for text in retrieved_texts[:k]:
        if is_relevant(text, golden_entry):
            return 1.0
    return 0.0


def precision_at_k(retrieved_texts: list[str], golden_entry: dict, k: int) -> float:
    """Fraction of top-k chunks that are relevant."""
    relevant = sum(1 for text in retrieved_texts[:k] if is_relevant(text, golden_entry))
    return relevant / k if k > 0 else 0.0


def mrr(retrieved_texts: list[str], golden_entry: dict) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant chunk."""
    for i, text in enumerate(retrieved_texts):
        if is_relevant(text, golden_entry):
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(retrieved_texts: list[str], golden_entry: dict, k: int) -> float:
    """1.0 if any of top-K chunks is relevant, else 0.0."""
    return recall_at_k(retrieved_texts, golden_entry, k)


def compute_all_metrics(
    all_results: list[list[dict]],
    golden_dataset: list[dict],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Compute all 6 metrics across all queries.

    Args:
        all_results: List of lists, each inner list contains
                     {"id", "text", "score"} dicts from Azure AI Search.
        golden_dataset: List of golden dataset entries.
        k_values: K values for Recall@K, Precision@K, etc.

    Returns:
        Dictionary with all aggregated metrics.
    """
    metrics = {}

    for k in k_values:
        recalls = [recall_at_k([r["text"] for r in res], golden, k)
                   for res, golden in zip(all_results, golden_dataset)]
        precisions = [precision_at_k([r["text"] for r in res], golden, k)
                      for res, golden in zip(all_results, golden_dataset)]
        hits = [hit_rate_at_k([r["text"] for r in res], golden, k)
                for res, golden in zip(all_results, golden_dataset)]
        ndcgs = [ndcg_at_k([r["text"] for r in res], golden, k)
                 for res, golden in zip(all_results, golden_dataset)]

        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"precision@{k}"] = float(np.mean(precisions))
        metrics[f"hit_rate@{k}"] = float(np.mean(hits)) * 100.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))

    mrrs = [mrr([r["text"] for r in res], golden)
            for res, golden in zip(all_results, golden_dataset)]
    metrics["mrr"] = float(np.mean(mrrs))
    metrics["total_questions"] = len(golden_dataset)

    return metrics
