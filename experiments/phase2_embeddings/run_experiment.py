"""
Phase 2: Embedding Model Evaluation
Run all Azure OpenAI embedding models against the golden dataset using Azure AI Search.
"""
import sys
import json
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import GOLDEN_DATASET_PATH
from src.utils.document_loader import load_all_documents, load_golden_dataset
from src.chunking.recursive import recursive_chunks
from src.embeddings.azure_openai import get_embeddings, get_query_embedding
from src.retrieval.azure_ai_search import (
    create_index,
    index_chunks,
    vector_search,
    delete_all_documents,
)
from src.evaluation.cost_metrics import CostTracker, LatencyTracker

from config import (
    EMBEDDING_MODELS,
    STRATEGY_PREFIX,
    RETRIEVAL_TOP_K,
    K_VALUES,
    EMBEDDING_BATCH_SIZE,
    RESULTS_DIR,
)
from evaluation.retrieval_metrics import compute_all_metrics
from evaluation.cosine_analysis import compute_cosine_distributions


def run_single_model(model_name: str, model_config: dict) -> dict:
    """
    Run full evaluation for one Azure OpenAI embedding model.

    Steps:
    1. Load recursive chunks from Phase 1
    2. Load golden dataset (50 questions)
    3. Recreate Azure AI Search index with correct dimension
    4. Embed all chunks (batch)
    5. Index chunks into Azure AI Search
    6. For each of 50 golden questions: embed query, vector search, evaluate
    7. Aggregate metrics, compute cosine distributions
    8. Track cost + latency
    9. Delete model's documents from index
    10. Return results
    """
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"  Dimension: {model_config['dimension']}")
    print(f"  Cost/1M tokens: ${model_config['cost_per_million_tokens']}")
    print(f"  Deployment: {model_config['deployment']}")
    print(f"{'=' * 60}")

    deployment = model_config["deployment"]
    dimension = model_config["dimension"]
    strategy_filter = f"{STRATEGY_PREFIX}_{model_name.replace('text-embedding-', '').replace('-', '_')}"

    cost_tracker = CostTracker(cost_per_million=model_config["cost_per_million_tokens"])
    latency = LatencyTracker()

    # --- Step 1-2: Load chunks and golden dataset ---
    print("\n[1/7] Loading chunks and golden dataset...")
    t0 = time.time()
    pages = load_all_documents()
    chunks = recursive_chunks(pages, chunk_size=512, chunk_overlap=50)
    golden = load_golden_dataset()
    latency.chunking_ms = (time.time() - t0) * 1000
    print(f"  Loaded {len(chunks)} chunks, {len(golden)} questions")

    # --- Step 3: Recreate index ---
    print(f"\n[2/7] Creating Azure AI Search index (dim={dimension})...")
    t0 = time.time()
    create_index(dimension=dimension)
    latency.indexing_ms = (time.time() - t0) * 1000

    # --- Step 4: Embed all chunks ---
    print(f"\n[3/7] Embedding {len(chunks)} chunks...")
    t0 = time.time()
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = get_embeddings(chunk_texts, batch_size=EMBEDDING_BATCH_SIZE, deployment=deployment)
    latency.embedding_generation_ms = (time.time() - t0) * 1000

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    chunk_tokens = sum(len(enc.encode(t)) for t in chunk_texts)
    cost_tracker.add_embedding_batch(chunk_tokens, len(chunk_texts))
    print(f"  Embedded {len(chunks)} chunks in {latency.embedding_generation_ms / 1000:.1f}s")

    # --- Step 5: Index chunks ---
    print(f"\n[4/7] Indexing chunks into Azure AI Search...")
    t0 = time.time()
    index_chunks(chunks, chunk_embeddings, strategy=strategy_filter)
    latency.indexing_ms += (time.time() - t0) * 1000

    # --- Step 6: Retrieve and evaluate per query ---
    print(f"\n[5/7] Retrieving top-{RETRIEVAL_TOP_K} for {len(golden)} queries...")
    t0 = time.time()
    all_results = []
    per_query_recall5 = []

    for i, q in enumerate(golden):
        query_emb_start = time.time()
        results = vector_search(
            q["question"],
            top_k=RETRIEVAL_TOP_K,
            strategy_filter=strategy_filter,
            deployment_name=deployment,
        )
        all_results.append(results)

        r5 = 1.0 if any(
            q["supporting_text"].lower() in r["text"].lower()
            for r in results[:5]
        ) else 0.0
        per_query_recall5.append(r5)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(golden)} queries...")

    latency.retrieval_ms = (time.time() - t0) * 1000

    # vector_search() embeds each query individually: 1 API call per query.
    query_tokens = sum(len(enc.encode(q["question"])) for q in golden)
    cost_tracker.add_embedding_batch(query_tokens, len(golden), api_calls=len(golden))

    # --- Step 7: Compute metrics ---
    print(f"\n[6/7] Computing metrics...")
    t0 = time.time()
    metrics = compute_all_metrics(all_results, golden, k_values=K_VALUES)
    cosine_dist = compute_cosine_distributions(all_results, golden)
    latency.evaluation_ms = (time.time() - t0) * 1000

    # --- Cleanup ---
    print(f"\n[7/7] Cleaning up index...")
    t0 = time.time()
    delete_all_documents(strategy_filter=strategy_filter)
    latency.total_ms = sum([
        latency.chunking_ms,
        latency.embedding_generation_ms,
        latency.indexing_ms,
        latency.retrieval_ms,
        latency.evaluation_ms,
    ])

    # --- Print summary ---
    print(f"\n{'─' * 50}")
    print(f"  RESULTS: {model_name}")
    print(f"{'─' * 50}")
    for k in K_VALUES:
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print(f"  MRR:      {metrics['mrr']:.4f}")
    for k in K_VALUES:
        print(f"  NDCG@{k}:  {metrics[f'ndcg@{k}']:.4f}")
    print(f"  Cosine separation: {cosine_dist['cosine_separation']:.4f}")
    print(f"  Cost: ${cost_tracker.summary()['embedding_cost_usd']:.6f}")
    print(f"  Total time: {latency.total_ms / 1000:.1f}s")

    # --- Build result ---
    result = {
        "model": model_name,
        "deployment": deployment,
        "dimension": dimension,
        "cost_per_million_tokens": model_config["cost_per_million_tokens"],
        "chunk_strategy": "recursive",
        "chunk_count": len(chunks),
        "strategy_filter": strategy_filter,
        "retrieval_metrics": metrics,
        "cosine_distributions": cosine_dist,
        "per_query_recall@5": per_query_recall5,
        "cost": cost_tracker.summary(),
        "latency": latency.summary(),
    }

    return result


def save_result(result: dict, output_dir: Path):
    """Save a single model result to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = result["model"].replace("text-embedding-", "").replace("-", "_")
    path = output_dir / f"{model_safe}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")


def run_all_models():
    """Run all 3 models sequentially, then comparison analysis."""
    print("=" * 60)
    print("  Phase 2: Embedding Model Evaluation")
    print("  Azure AI Search + Azure OpenAI Embeddings")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name, model_config in EMBEDDING_MODELS.items():
        try:
            result = run_single_model(model_name, model_config)
            save_result(result, RESULTS_DIR)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNo models completed successfully.")
        return

    # --- Comparison analysis ---
    print(f"\n{'=' * 60}")
    print("  COMPARISON ANALYSIS")
    print(f"{'=' * 60}")

    from evaluation.comparison import compare_models
    comparison = compare_models(all_results)

    comparison_path = RESULTS_DIR / "comparison_summary.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  Saved: {comparison_path}")

    # --- Print comparison table ---
    print(f"\n{'─' * 70}")
    print(f"  {'Model':<30} {'R@5':>6} {'MRR':>6} {'NDCG@5':>7} {'Cost/1M':>9} {'Composite':>10}")
    print(f"{'─' * 70}")
    for entry in comparison["ranking"]:
        print(f"  {entry['model']:<30} {entry['recall@5']:>6.4f} {entry['mrr']:>6.4f} "
              f"{entry['ndcg@5']:>7.4f} ${entry['cost_per_million']:>7.2f} {entry['composite_score']:>10.4f}")
    print(f"{'─' * 70}")

    print(f"\n  Winner: {comparison['best_overall']}")
    print(f"  Best Recall@5: {comparison['best_recall@5']['model']} = {comparison['best_recall@5']['value']:.4f}")

    if comparison.get("statistical_significance"):
        print(f"\n  Statistical Significance (vs baseline):")
        for pair, test in comparison["statistical_significance"].items():
            sig = "YES" if test["significant"] else "no"
            print(f"    {pair}: p={test['p_value']:.4f} [{sig}]")

    # --- Generate visualizations ---
    print(f"\n{'=' * 60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'=' * 60}")

    try:
        from visualization.metric_charts import generate_metric_charts
        generate_metric_charts(all_results, RESULTS_DIR)
    except Exception as e:
        print(f"  Warning: metric_charts failed: {e}")

    try:
        from visualization.cosine_distributions import generate_cosine_plots
        generate_cosine_plots(all_results, RESULTS_DIR)
    except Exception as e:
        print(f"  Warning: cosine_distributions failed: {e}")

    try:
        from visualization.cost_quality_scatter import generate_cost_quality_plot
        generate_cost_quality_plot(all_results, comparison, RESULTS_DIR)
    except Exception as e:
        print(f"  Warning: cost_quality_scatter failed: {e}")

    print(f"\n{'=' * 60}")
    print("  PHASE 2 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_all_models()
