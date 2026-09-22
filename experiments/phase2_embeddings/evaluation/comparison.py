import numpy as np
from scipy import stats


def compare_models(results: list[dict]) -> dict:
    """
    Cross-model comparison with statistical significance tests.

    Args:
        results: List of per-model result dicts (from run_single_model).

    Returns:
        Comparison summary with rankings, cost analysis, significance tests.
    """
    model_names = [r["model"] for r in results]

    # --- Rankings ---
    rankings = []
    for r in results:
        metrics = r["retrieval_metrics"]
        rankings.append({
            "model": r["model"],
            "recall@5": metrics.get("recall@5", 0),
            "mrr": metrics.get("mrr", 0),
            "ndcg@5": metrics.get("ndcg@5", 0),
            "cost_per_million": r["cost_per_million_tokens"],
        })

    # Composite score: 0.30*recall@5 + 0.25*mrr + 0.20*ndcg@5 + 0.15*sep + 0.10*cost_eff
    costs = [r["cost_per_million_tokens"] for r in results]
    max_cost = max(costs) if costs else 1.0

    for entry in rankings:
        model_result = next(r for r in results if r["model"] == entry["model"])
        cosine_sep = model_result.get("cosine_distributions", {}).get("cosine_separation", 0)
        cost_eff = 1.0 - (entry["cost_per_million"] / max_cost) if max_cost > 0 else 1.0

        entry["cosine_separation"] = cosine_sep
        entry["cost_efficiency"] = cost_eff
        entry["composite_score"] = (
            0.30 * entry["recall@5"]
            + 0.25 * entry["mrr"]
            + 0.20 * entry["ndcg@5"]
            + 0.15 * cosine_sep
            + 0.10 * cost_eff
        )

    rankings.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, entry in enumerate(rankings):
        entry["rank"] = i + 1

    # --- Statistical significance (paired t-test on per-query recall@5) ---
    significance = {}
    baseline_name = results[0]["model"] if results else None
    baseline_queries = results[0].get("per_query_recall@5", []) if results else []

    for r in results[1:]:
        model_name = r["model"]
        model_queries = r.get("per_query_recall@5", [])
        if len(baseline_queries) == len(model_queries) and len(baseline_queries) > 1:
            t_stat, p_value = stats.ttest_rel(model_queries, baseline_queries)
            significance[f"{model_name}_vs_{baseline_name}"] = {
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }

    # --- Cost-quality analysis ---
    cost_quality = {}
    for r in results:
        m = r["retrieval_metrics"]
        cost_per_1k = (r["cost_per_million_tokens"] / 1_000_000) * 1000 * 150
        recall = m.get("recall@5", 0)
        cost_quality[r["model"]] = {
            "cost_per_1k_queries": round(cost_per_1k, 4),
            "recall@5": recall,
            "quality_per_dollar": round(recall / cost_per_1k, 2) if cost_per_1k > 0 else float("inf"),
        }

    best_composite = rankings[0] if rankings else {}
    best_recall = max(rankings, key=lambda x: x["recall@5"]) if rankings else {}
    best_cost_eff = min(results, key=lambda x: x["cost_per_million_tokens"]) if results else {}

    return {
        "models_tested": len(results),
        "best_overall": best_composite.get("model"),
        "best_recall@5": {"model": best_recall.get("model"), "value": best_recall.get("recall@5")},
        "best_cost_efficiency": {"model": best_cost_eff.get("model")},
        "ranking": rankings,
        "statistical_significance": significance,
        "cost_quality_analysis": cost_quality,
    }
