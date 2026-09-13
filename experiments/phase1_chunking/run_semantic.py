"""
Evaluate semantic chunking with Azure AI Search + cost/latency tracking.
NOTE: Slow - embeds every sentence to find split points.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.document_loader import load_all_documents, load_golden_dataset
from src.chunking.semantic import semantic_chunks
from src.embeddings.azure_openai import get_embeddings
from src.retrieval.azure_ai_search import create_index, index_chunks, vector_search, delete_all_documents
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.cost_metrics import compute_index_stats, ExperimentResult, LatencyTracker, CostTracker
import tiktoken


def run():
    print("=" * 50)
    print("Semantic Chunking + Azure AI Search")
    print("=" * 50)
    print("NOTE: Slow - embeds every sentence to find split points.\n")

    latency = LatencyTracker()
    cost = CostTracker()
    total_start = time.time()

    pages = load_all_documents()
    golden = load_golden_dataset()
    print(f"Pages: {len(pages)} | Questions: {len(golden)}\n")

    print("Chunking (this takes a while)...", flush=True)
    t0 = time.time()
    chunks = semantic_chunks(pages, chunk_size=512, breakpoint_threshold=75.0)
    latency.chunking_ms = (time.time() - t0) * 1000
    print(f"Chunking done: {latency.chunking_ms:.0f}ms", flush=True)

    stats = compute_index_stats(chunks)
    print(f"Chunks: {stats['chunk_count']} | Avg tokens: {stats['avg_chunk_tokens']:.0f}")

    print("Embedding chunks...", flush=True)
    t0 = time.time()
    embeddings = get_embeddings([c.text for c in chunks])
    latency.embedding_generation_ms = (time.time() - t0) * 1000

    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(c.text)) for c in chunks)
    cost.add_embedding_batch(total_tokens, len(chunks))
    print(f"Embeddings: {latency.embedding_generation_ms:.0f}ms | Tokens: {total_tokens} | Cost: ${cost.embedding_cost_usd:.4f}")

    print("Indexing to Azure AI Search...", flush=True)
    create_index(dimension=1536)
    index_chunks(chunks, embeddings, strategy="semantic")

    print("Retrieving & evaluating...", flush=True)
    t0 = time.time()
    all_retrieved = []
    for q in golden:
        results = vector_search(q["question"], top_k=5, strategy_filter="semantic")
        class R:
            def __init__(self, text): self.text = text
        all_retrieved.append([R(r["text"]) for r in results])
    latency.retrieval_ms = (time.time() - t0) * 1000

    t0 = time.time()
    metrics = compute_retrieval_metrics(all_retrieved, golden, k_values=[1, 3, 5])
    latency.evaluation_ms = (time.time() - t0) * 1000
    latency.total_ms = (time.time() - total_start) * 1000

    print("\n--- Retrieval Metrics ---")
    for k, v in metrics.items(): print(f"  {k}: {v:.3f}")
    print("\n--- Cost ---")
    for k, v in cost.summary().items(): print(f"  {k}: {v}")
    print("\n--- Latency ---")
    for k, v in latency.summary().items(): print(f"  {k}: {v}")

    result = ExperimentResult(
        strategy="semantic", config={"breakpoint_threshold": 75.0},
        chunk_count=stats["chunk_count"], avg_chunk_tokens=stats["avg_chunk_tokens"],
        retrieval_metrics=metrics, cost=cost, latency=latency,
    )
    result.save(Path(__file__).parent / "results" / "run_2")
    delete_all_documents(strategy_filter="semantic")


if __name__ == "__main__":
    run()
