# RAG Evaluation & Optimization Lab

## Objective
Determine which RAG architecture works best for your documents, and why.

## Golden Dataset
- **Questions**: 50 (fact, cause_effect, comparison, list, multi_concept, boundary_sensitive, table_lookup, context, multi_sentence, structured_data, definition_and_multi_step_reasoning)
- **Source**: P&G Annual Report 2025
- **Location**: `data/pg_2025_gold_evaluation_dataset.json`

---

## Phase 1: Chunking Experiments (CURRENT PHASE)

**Goal**: Compare 5 chunking strategies on the same document, measure impact on retrieval quality.

| Strategy | Description |
|----------|-------------|
| Fixed-size | 512 tokens, 50 overlap |
| Recursive | Split by paragraphs/sections recursively |
| Semantic | Group semantically similar sentences |
| Markdown/structure-aware | Respect headings, lists, tables |
| Parent-child | Small retrieval chunks with parent context |

### Metrics (Phase 1)
- Chunk count, avg chunk size, tokens per chunk
- Retrieval recall@k (k=1,3,5) against golden dataset
- Retrieval precision@k
- MRR (Mean Reciprocal Rank)

---

## Phase 2: Embedding Model Comparison

**Goal**: Test multiple embedding models on best chunking strategy from Phase 1.

| Model | Provider |
|-------|----------|
| text-embedding-3-large | Azure OpenAI |
| text-embedding-3-small | Azure OpenAI |
| Cohere embed-v3 | Azure/Cohere |
| BGE-large-en-v1.5 | Open-source (local) |

### Metrics (Phase 2)
- Retrieval accuracy (recall, precision, MRR)
- Embedding generation latency
- Cost per 1M tokens

---

## Phase 3: Retrieval Strategy Comparison

**Goal**: Test different retrieval approaches with best chunks + embeddings from Phase 1 & 2.

| Strategy | Description |
|----------|-------------|
| Vector search | Pure semantic similarity |
| BM25 | Keyword-based sparse retrieval |
| Hybrid search | Vector + BM25 combined |
| Reranking | Vector search + cross-encoder reranker |

### Metrics (Phase 3)
- Retrieval quality (recall, precision, MRR, NDCG)
- Latency per query
- End-to-end answer quality (faithfulness, relevancy)

---

## Phase 4: LLM Generation Evaluation

**Goal**: Compare LLM outputs across configurations using golden dataset ground truth.

### Metrics (Phase 4)
- Faithfulness (answer grounded in context)
- Relevancy (answer addresses question)
- Answer correctness (vs ground truth)
- Hallucination rate
- Token usage & cost

---

## Phase 5: Cost & Latency Optimization

**Goal**: Find optimal balance between quality and cost.

### Metrics (Phase 5)
- Total cost per query
- End-to-end latency
- Quality/cost ratio

---

## Azure Services Used

| Service | Purpose |
|---------|---------|
| Azure OpenAI (gpt-4o, gpt-4o-mini) | LLM generation |
| Azure OpenAI Embeddings | text-embedding-3-large/small |
| Azure AI Search | Vector search, BM25, hybrid |
| Azure Blob Storage | Document storage |
| Azure Cosmos DB | Experiment results storage |
| Azure ML / Prompt Flow | Evaluation pipelines |

---

## Folder Structure

```
RAGEvaluation&OptimizationLab/
├── PROJECT_PLAN.md                 # This file - master plan
├── README.md                       # Project overview
├── .env.example                    # Azure credentials template
├── requirements.txt
├── data/
│   ├── pg_2025_gold_evaluation_dataset.json   # Golden dataset (50 Qs)
│   └── source_documents/                      # PDFs to index
├── docs/
│   ├── PHASE_1_CHUNKING.md         # Chunking experiment docs
│   ├── PHASE_2_EMBEDDINGS.md       # Embedding model docs
│   ├── PHASE_3_RETRIEVAL.md        # Retrieval strategy docs
│   ├── PHASE_4_GENERATION.md       # LLM generation docs
│   └── PHASE_5_OPTIMIZATION.md     # Cost/latency docs
├── src/
│   ├── __init__.py
│   ├── config.py                   # Azure config, paths, constants
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── fixed_size.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   ├── markdown_aware.py
│   │   └── parent_child.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── azure_openai.py
│   │   └── local_models.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   ├── bm25_search.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   └── cost_metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── azure_client.py
│       └── document_loader.py
├── experiments/
│   ├── phase1_chunking/
│   │   ├── run_experiment.py
│   │   ├── results/
│   │   └── analysis.ipynb
│   └── ...future phases
└── notebooks/
    └── exploration.ipynb
```

---

## Execution Order

1. Set up Azure resources and `.env`
2. Place source PDFs in `data/source_documents/`
3. **Phase 1**: Run chunking experiments → pick best strategy
4. **Phase 2**: Run embedding experiments → pick best model
5. **Phase 3**: Run retrieval experiments → pick best strategy
6. **Phase 4**: Run generation experiments → pick best LLM
7. **Phase 5**: Optimize for cost/latency
8. Final recommendation report
