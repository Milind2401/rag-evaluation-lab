import numpy as np


def compute_cosine_distributions(
    all_results: list[list[dict]],
    golden_dataset: list[dict],
) -> dict:
    """
    Analyze cosine similarity distributions from Azure AI Search @search.score.

    Args:
        all_results: List of lists, each inner list contains
                     {"id", "text", "score"} dicts from Azure AI Search.
        golden_dataset: List of golden dataset entries.

    Returns:
        Dictionary with distribution statistics.
    """
    all_scores = []
    relevant_scores = []
    irrelevant_scores = []

    for results, golden in zip(all_results, golden_dataset):
        supporting_text = golden.get("supporting_text", "").lower()
        for r in results:
            score = r["score"]
            all_scores.append(score)
            if supporting_text and supporting_text in r["text"].lower():
                relevant_scores.append(score)
            else:
                irrelevant_scores.append(score)

    all_scores = np.array(all_scores) if all_scores else np.array([0.0])
    relevant_scores = np.array(relevant_scores) if relevant_scores else np.array([0.0])
    irrelevant_scores = np.array(irrelevant_scores) if irrelevant_scores else np.array([0.0])

    def _stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "count": len(arr),
        }

    mean_rel = float(np.mean(relevant_scores)) if len(relevant_scores) > 0 else 0.0
    mean_irrel = float(np.mean(irrelevant_scores)) if len(irrelevant_scores) > 0 else 0.0

    return {
        "overall": _stats(all_scores),
        "relevant": _stats(relevant_scores),
        "irrelevant": _stats(irrelevant_scores),
        "cosine_separation": mean_rel - mean_irrel,
        "avg_cosine_relevant": mean_rel,
        "avg_cosine_irrelevant": mean_irrel,
    }
