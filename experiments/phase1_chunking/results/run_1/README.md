# Phase 1 — Run 1 Results

## Date: 2026-09-10

## Configuration

| Parameter | Value |
|-----------|-------|
| Document | P&G_AnnualReport.pdf (converted to markdown) |
| Pages | 301 |
| Golden Questions | 50 |
| Embedding Model | text-embedding-3-small (Azure OpenAI) |
| Embedding Dimension | 1536 |
| Retrieval | Azure AI Search (vector search, top-5) |
| API Version | 2023-05-15 |

## Chunking Strategies Tested

| Strategy | Config | Chunks | Avg Tokens |
|----------|--------|--------|------------|
| fixed_size | chunk_size=512, overlap=50 | 405 | 225 |
| recursive | chunk_size=512, overlap=50 | 1269 | 68 |
| markdown_aware | max_chunk_size=1024 | 602 | 143 |
| parent_child | parent=1024, child=256, overlap=30 | 2179 | 41 |
| semantic | breakpoint_threshold=75.0 | 742 | 116 |

## Results

| Strategy | Recall@1 | Recall@3 | Recall@5 | Precision@3 | MRR | Cost | Time |
|----------|----------|----------|----------|-------------|-----|------|------|
| **fixed_size** | 0.22 | **0.38** | 0.38 | 0.127 | **0.293** | $0.0018 | 235s |
| markdown_aware | 0.20 | 0.34 | 0.36 | 0.113 | 0.264 | $0.0017 | 190s |
| semantic | 0.12 | 0.28 | 0.30 | 0.093 | 0.194 | $0.0017 | 300s |
| recursive | 0.14 | 0.26 | 0.34 | 0.087 | 0.216 | $0.0017 | 214s |
| parent_child | 0.14 | 0.22 | 0.22 | 0.073 | 0.180 | $0.0018 | 240s |

## Winner: fixed_size

- **Recall@3: 38%** — 19 out of 50 questions had relevant chunk in top-3
- **MRR: 0.293** — first relevant result appears at rank ~3.4 on average
- **Best balance** of quality, cost, and latency

## Key Findings

1. **Larger chunks perform better** — fixed_size (225 tokens) beat parent_child (41 tokens) by 16 points
2. **Small chunks lose context** — golden supporting_text spans across multiple chunks when chunks are too small
3. **Markdown-aware is second best** — respects document structure but creates uneven chunks
4. **Semantic is too slow** — 111s chunking time vs <1s for others
5. **Cost is identical** — all strategies embedded ~85-91K tokens = ~$0.0017
6. **Retrieval time dominates** — 150-160s for 50 queries (Azure AI Search latency)

## Recall@3 vs Chunk Size

| Strategy | Avg Tokens | Recall@3 |
|----------|------------|----------|
| fixed_size | 225 | 38% |
| markdown_aware | 143 | 34% |
| semantic | 116 | 28% |
| recursive | 68 | 26% |
| parent_child | 41 | 22% |

**Trend: bigger chunks = better retrieval**

## What Didn't Work

- **parent_child**: Child chunks too small (41 tokens), supporting_text gets split
- **recursive**: Over-split into 1269 chunks, loses context
- **semantic**: Slow and mid-sized chunks don't align with answer boundaries

## Next Steps

- Proceed to Phase 2 with **fixed_size** strategy
- Test different chunk sizes: 256, 512, 1024
- Test different overlaps: 0, 50, 100
- Compare embedding models (text-embedding-3-large, BGE)

---

## Azure Portal Metrics (from screenshots)

**Azure OpenAI:**
| Metric | Value |
|--------|-------|
| Input Tokens | 1.03M |
| Model Requests | 530 |
| Avg Latency | 269.78ms |

**Azure AI Search:**
| Metric | Value |
|--------|-------|
| Search Queries/sec | 1.26 |
| Search Latency | 13.85ms |
| Vector Storage | 6MB |

**Code-Level Cost:**
| Metric | Value |
|--------|-------|
| Embedding Cost | ~$0.0018 per strategy |
| Total Run Time | 190-300s per strategy |

---

## Evaluation Limitations (Run 1)

### Method Used: Substring Match

```python
supporting_text.lower() in retrieved_chunk.text.lower()
```

Checks if the exact `supporting_text` from the golden dataset appears as a substring in the retrieved chunk.

### Limitations

| Issue | Impact |
|-------|--------|
| **Exact text only** | Chunk with same meaning but different wording scores 0 |
| **Markdown conversion** | Text changed from PDF → markdown, may not match exactly |
| **No partial credit** | 90% match scores same as 0% match |
| **No semantic understanding** | Doesn't know if chunk is actually useful for answering |
| **Binary scoring** | Only 0 or 1 — no gradient of relevance |

### What Run 2 Fixes

Run 2 uses **semantic similarity** instead of substring match:
- Embeds both the `supporting_text` and the retrieved chunk
- Calculates cosine similarity between them
- Scores based on semantic closeness, not exact text
- Provides partial credit for semantically similar content

---

## Screenshots

| File | Description |
|------|-------------|
| Screenshot 2026-09-10 140445.png | Azure OpenAI - Input Tokens |
| Screenshot 2026-09-10 140815.png | Azure OpenAI - Model Requests |
| Screenshot 2026-09-10 141013.png | Azure OpenAI - Avg Latency |
| Screenshot 2026-09-10 141201.png | Azure AI Search - Queries/sec |
| Screenshot 2026-09-10 141303.png | Azure AI Search - Vector Storage |
| Screenshot 2026-09-10 141315.png | Azure AI Search - Search Latency |
