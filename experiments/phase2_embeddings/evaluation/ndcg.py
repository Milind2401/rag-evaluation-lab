import numpy as np


def dcg_at_k(relevance: list[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevance = np.array(relevance, dtype=float)[:k]
    if len(relevance) == 0:
        return 0.0
    return float(np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2))))


def ndcg_at_k(retrieved_texts: list[str], golden_entry: dict, k: int) -> float:
    """Normalized DCG@K for binary relevance."""
    supporting_text = golden_entry.get("supporting_text", "").lower()

    relevance = []
    for text in retrieved_texts[:k]:
        if supporting_text and supporting_text in text.lower():
            relevance.append(1)
        else:
            relevance.append(0)

    actual_dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
