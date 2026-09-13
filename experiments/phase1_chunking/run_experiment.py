"""
Phase 1: Chunking Experiments
Run all chunking strategies against the golden dataset and compare retrieval quality.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import GOLDEN_DATASET_PATH, SOURCE_DOCS_DIR
from src.utils.document_loader import load_all_documents, load_golden_dataset
from src.chunking import CHUNKING_FUNCTIONS
from src.embeddings.azure_openai import get_embeddings, get_query_embedding, cosine_similarity
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.cost_metrics import compute_index_stats, ExperimentResult
import numpy as np


def embedding_search(chunks, query: str, embeddings_cache: np.ndarray, top_k: int = 5):
    """Retrieve top-k chunks using cosine similarity with embeddings."""
    query_emb = get_query_embedding(query)
    similarities = [cosine_similarity(query_emb, emb) for emb in embeddings_cache]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def run_experiment():
    print("=" * 60)
    print("Phase 1: Chunking Experiments (Embedding Retrieval)")
    print("=" * 60)

    pdfs = list(SOURCE_DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"\nNo PDF files found in {SOURCE_DOCS_DIR}")
        print("Please place your source PDF(s) there and re-run.")
        return

    print(f"\nFound {len(pdfs)} document(s): {[p.name for p in pdfs]}")

    pages = load_all_documents()
    golden = load_golden_dataset()
    print(f"Loaded {len(pages)} pages, {len(golden)} golden questions")

    results = []
    # Skip semantic for now - too slow (embeds every sentence)
    strategies = {k: v for k, v in CHUNKING_FUNCTIONS.items() if k != "semantic"}

    for strategy_name, chunk_fn in strategies.items():
        print(f"\n--- Running: {strategy_name} ---", flush=True)

        kwargs = {"chunk_size": 512, "chunk_overlap": 50}
        if strategy_name == "parent_child":
            kwargs = {"parent_size": 1024, "child_size": 256, "child_overlap": 30}
        elif strategy_name == "semantic":
            kwargs = {"chunk_size": 512, "breakpoint_threshold": 75.0}
        elif strategy_name == "markdown_aware":
            kwargs = {"max_chunk_size": 1024}

        try:
            start = time.time()
            chunks = chunk_fn(pages, **kwargs)
            index_time = time.time() - start

            stats = compute_index_stats(chunks)
            print(f"  Chunks: {stats['chunk_count']}, Avg tokens: {stats['avg_chunk_tokens']:.0f}", flush=True)

            # Generate embeddings for all chunks
            print(f"  Generating embeddings for {len(chunks)} chunks...", flush=True)
            embed_start = time.time()
            chunk_texts = [c.text for c in chunks]
            embeddings_cache = np.array(get_embeddings(chunk_texts))
            embed_time = time.time() - embed_start
            print(f"  Embeddings generated in {embed_time:.1f}s", flush=True)

            # Retrieve for each golden question
            all_retrieved = []
            for q in golden:
                retrieved = embedding_search(chunks, q["question"], embeddings_cache, top_k=5)
                all_retrieved.append(retrieved)

            metrics = compute_retrieval_metrics(all_retrieved, golden, k_values=[1, 3, 5])
            print(f"  Recall@3: {metrics['recall@3']:.3f}, MRR: {metrics['mrr']:.3f}")

            result = ExperimentResult(
                strategy=strategy_name,
                chunk_size=kwargs.get("chunk_size", kwargs.get("parent_size", 0)),
                overlap=kwargs.get("chunk_overlap", kwargs.get("child_overlap", 0)),
                chunk_count=stats["chunk_count"],
                avg_chunk_tokens=stats["avg_chunk_tokens"],
                index_time_seconds=index_time,
                retrieval_metrics=metrics,
                embedding_latency_ms=embed_time * 1000 / len(chunks),
            )
            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(__file__).parent / "results"
    output_path.mkdir(exist_ok=True)

    results_data = [r.to_dict() for r in results]
    with open(output_path / "phase1_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r.strategy}:")
        print(f"  Chunks: {r.chunk_count}, Avg tokens: {r.avg_chunk_tokens:.0f}")
        for k, v in r.retrieval_metrics.items():
            print(f"  {k}: {v:.3f}")

    if results:
        best = max(results, key=lambda x: x.retrieval_metrics.get("recall@3", 0))
        print(f"\n{'=' * 60}")
        print(f"BEST STRATEGY: {best.strategy} (Recall@3: {best.retrieval_metrics['recall@3']:.3f})")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    run_experiment()
