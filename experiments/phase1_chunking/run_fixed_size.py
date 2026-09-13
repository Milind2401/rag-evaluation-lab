"""
Evaluate fixed_size chunking with Azure AI Search + cost/latency tracking.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.document_loader import load_all_documents, load_golden_dataset
from src.chunking.fixed_size import fixed_size_chunks
from src.embeddings.azure_openai import get_embeddings
from src.retrieval.azure_ai_search import create_index, index_chunks, vector_search, delete_all_documents
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.cost_metrics import compute_index_stats, ExperimentResult, LatencyTracker, CostTracker
from src.config import GOLDEN_DATASET_PATH


def run():
    print("=" * 50)
    print("Fixed-Size Chunking + Azure AI Search")
    print("=" * 50)

    latency = LatencyTracker()
    cost = CostTracker()
    total_start = time.time()

    # 1. Load
    pages = load_all_documents()
    golden = load_golden_dataset()
    print(f"Pages: {len(pages)} | Questions: {len(golden)}\n")

    # 2. Chunk
    t0 = time.time()
    chunks = fixed_size_chunks(pages, chunk_size=512, chunk_overlap=50)
    latency.chunking_ms = (time.time() - t0) * 1000

    stats = compute_index_stats(chunks)
    print(f"Chunks: {stats['chunk_count']} | Avg tokens: {stats['avg_chunk_tokens']:.0f}")

    # 3. Embed chunks
    print("Embedding chunks...", flush=True)
    t0 = time.time()
    chunk_texts = [c.text for c in chunks]
    embeddings = get_embeddings(chunk_texts)
    latency.embedding_generation_ms = (time.time() - t0) * 1000

    # Track embedding cost
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(t)) for t in chunk_texts)
    cost.add_embedding_batch(total_tokens, len(chunk_texts))
    print(f"Embeddings: {latency.embedding_generation_ms:.0f}ms | Tokens: {total_tokens} | Cost: ${cost.embedding_cost_usd:.4f}")

    # 4. Index to Azure AI Search
    print("Indexing to Azure AI Search...", flush=True)
    create_index(dimension=1536)
    index_chunks(chunks, embeddings, strategy="fixed_size")

    # 5. Retrieve + Evaluate
    print("Retrieving & evaluating...", flush=True)
    t0 = time.time()
    all_retrieved = []
    for q in golden:
        results = vector_search(q["question"], top_k=5, strategy_filter="fixed_size")
        # Convert to chunk-like objects for metrics
        class RetrievalResult:
            def __init__(self, text):
                self.text = text
        all_retrieved.append([RetrievalResult(r["text"]) for r in results])
    latency.retrieval_ms = (time.time() - t0) * 1000

    t0 = time.time()
    metrics = compute_retrieval_metrics(all_retrieved, golden, k_values=[1, 3, 5])
    latency.evaluation_ms = (time.time() - t0) * 1000

    latency.total_ms = (time.time() - total_start) * 1000

    # 6. Print results
    print("\n--- Retrieval Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    print("\n--- Cost ---")
    for k, v in cost.summary().items():
        print(f"  {k}: {v}")

    print("\n--- Latency ---")
    for k, v in latency.summary().items():
        print(f"  {k}: {v}")

    # 7. Save
    result = ExperimentResult(
        strategy="fixed_size",
        config={"chunk_size": 512, "overlap": 50},
        chunk_count=stats["chunk_count"],
        avg_chunk_tokens=stats["avg_chunk_tokens"],
        retrieval_metrics=metrics,
        cost=cost,
        latency=latency,
    )
    result.save(Path(__file__).parent / "results" / "run_2")

    # Cleanup
    delete_all_documents(strategy_filter="fixed_size")


if __name__ == "__main__":
    run()
