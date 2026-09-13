# RAG Evaluation & Optimization Lab

Systematic evaluation of RAG architectures to determine what works best for your documents.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Azure credentials
cp .env.example .env
# Edit .env with your Azure keys

# 3. Place source PDF(s)
# Copy your PDFs into data/raw_pdf/

# 4. Run Phase 1 (Chunking)
python experiments/phase1_chunking/run_experiment.py
```

## Project Structure

```
├── data/                    # Golden dataset + source documents
├── docs/                    # Phase documentation (updated as project progresses)
├── src/                     # Reusable modules
│   ├── chunking/            # 5 chunking strategies
│   ├── evaluation/          # Metrics and scoring
│   ├── retrieval/           # Search strategies (Phase 3)
│   ├── generation/          # LLM generation (Phase 4)
│   └── utils/               # Azure clients, document loader
├── experiments/             # Experiment runners per phase
└── PROJECT_PLAN.md          # Master plan with all phases
```

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Chunking | 🔄 In Progress | Compare 5 chunking strategies |
| 2. Embeddings | ⏳ Pending | Compare embedding models |
| 3. Retrieval | ⏳ Pending | Compare search strategies |
| 4. Generation | ⏳ Pending | Evaluate LLM outputs |
| 5. Optimization | ⏳ Pending | Cost/latency tuning |

## Azure Services

- **Azure OpenAI**: GPT-4o (generation), text-embedding-3-small (embeddings)
- **Azure AI Search**: Vector search, BM25, hybrid
- **Azure Blob Storage**: Document storage
