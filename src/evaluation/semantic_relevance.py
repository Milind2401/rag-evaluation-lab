"""
Semantic relevance evaluation — uses embedding similarity instead of substring match.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.embeddings.azure_openai import get_embeddings, cosine_similarity


def is_relevant_semantic(
    chunk_text: str,
    supporting_text: str,
    threshold: float = 0.75,
    support_embedding: np.ndarray = None,
    cost_tracker=None,
) -> tuple[bool, float]:
    """
    Check if chunk is semantically relevant to the golden answer.
    Returns (is_relevant, similarity_score).
    """
    if support_embedding is not None:
        # Only embed the chunk (support already embedded)
        chunk_emb = get_embeddings([chunk_text])[0]
        score = cosine_similarity(chunk_emb, support_embedding)
        if cost_tracker:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = len(enc.encode(chunk_text))
            cost_tracker.add_embedding_batch(tokens, 1)
    else:
        embeddings = get_embeddings([chunk_text, supporting_text])
        score = cosine_similarity(embeddings[0], embeddings[1])
        if cost_tracker:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = len(enc.encode(chunk_text)) + len(enc.encode(supporting_text))
            cost_tracker.add_embedding_batch(tokens, 2)

    return score >= threshold, score


def avg_similarity(
    retrieved_texts: list[str],
    support_embedding: np.ndarray,
    cost_tracker=None,
) -> float:
    """Average similarity score of top-k chunks to supporting_text."""
    if not retrieved_texts:
        return 0.0

    chunk_embeddings = get_embeddings(retrieved_texts)
    if cost_tracker:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = sum(len(enc.encode(t)) for t in retrieved_texts)
        cost_tracker.add_embedding_batch(tokens, len(retrieved_texts))

    similarities = [cosine_similarity(emb, support_embedding) for emb in chunk_embeddings]
    return float(np.mean(similarities))


def compute_semantic_metrics(
    all_results: list[list[str]],
    golden_dataset: list[dict],
    k_values: list[int] = [1, 3, 5],
    threshold: float = 0.75,
    cost_tracker=None,
) -> dict:
    """Compute aggregated semantic retrieval metrics."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # Pre-embed all supporting texts (1 API call for all 50)
    supporting_texts = [g["supporting_text"] for g in golden_dataset]
    support_embeddings = get_embeddings(supporting_texts)
    if cost_tracker:
        tokens = sum(len(enc.encode(t)) for t in supporting_texts)
        cost_tracker.add_embedding_batch(tokens, len(supporting_texts))

    metrics = {}

    for k in k_values:
        recalls = []
        precisions = []
        for result_texts, support_emb in zip(all_results, support_embeddings):
            relevant = 0
            found = False
            for text in result_texts[:k]:
                chunk_emb = get_embeddings([text])[0]
                score = cosine_similarity(chunk_emb, support_emb)
                if cost_tracker:
                    tokens = len(enc.encode(text))
                    cost_tracker.add_embedding_batch(tokens, 1)
                if score >= threshold:
                    relevant += 1
                    found = True
            recalls.append(1.0 if found else 0.0)
            precisions.append(relevant / k if k > 0 else 0.0)

        metrics[f"semantic_recall@{k}"] = np.mean(recalls)
        metrics[f"semantic_precision@{k}"] = np.mean(precisions)

    # MRR
    mrrs = []
    for result_texts, support_emb in zip(all_results, support_embeddings):
        for i, text in enumerate(result_texts):
            chunk_emb = get_embeddings([text])[0]
            score = cosine_similarity(chunk_emb, support_emb)
            if cost_tracker:
                tokens = len(enc.encode(text))
                cost_tracker.add_embedding_batch(tokens, 1)
            if score >= threshold:
                mrrs.append(1.0 / (i + 1))
                break
        else:
            mrrs.append(0.0)
    metrics["semantic_mrr"] = np.mean(mrrs)

    # Avg similarity
    avg_sims = []
    for result_texts, support_emb in zip(all_results, support_embeddings):
        avg_sims.append(avg_similarity(result_texts, support_emb, cost_tracker))
    metrics["avg_similarity"] = np.mean(avg_sims)

    metrics["threshold"] = threshold
    metrics["total_questions"] = len(golden_dataset)

    return metrics
