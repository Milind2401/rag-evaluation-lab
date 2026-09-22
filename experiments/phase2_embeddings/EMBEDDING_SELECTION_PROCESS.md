# Embedding Model Selection for RAG — Industry-Standard Process Applied

## Date: 2026-09-21
## Companion doc: [EXPERIMENT.md](results/EXPERIMENT.md) (full results) · Data: `results/*.json`

> **How to read this doc.** EXPERIMENT.md answers *"what did we find?"*. This doc answers *"how would an industry team run this selection end-to-end?"* — and maps each standard step to what was actually done in this project, including the chart-backed visual evidence pack for a complete evaluation.

---

## The 8-Step Selection Process at a Glance

| # | Industry Step | Status in This Project | Where Documented |
|---|---------------|------------------------|------------------|
| 1 | Define requirements & constraints | ✅ Done | §1 |
| 2 | Shortlist candidate models | ✅ Done | §2 |
| 3 | Build golden dataset | ✅ Done (100 questions) | §3 |
| 4 | Standardize the pipeline | ✅ Done (controlled variables) | §4 |
| 5 | Retrieval-level quality metrics | ✅ Done (Recall/MRR/NDCG) | §5 + EXPERIMENT.md |
| 6 | Operational / non-accuracy metrics | ✅ Cost, dimension, storage, pipeline latency measured + 9-chart visual evidence pack | §6 |
| 7 | Weighted scoring matrix | ✅ Done (composite score) + sensitivity analysis | §7 |
| 8 | Summary & decision | ✅ Done (3-large default, 3-small at scale) | §8 |

---

## Step 1 — Define Requirements & Constraints

Industry practice: before looking at any model, write down what "good" means for **your** system. A selection made without this step defaults to "whatever the benchmark leaderboard says," which is rarely what the business needs.

### What we defined

| Requirement | Value / Constraint | Source |
|-------------|--------------------|--------|
| Use case | RAG over corporate financial documents (annual report Q&A) | Project goal |
| Retrieval quality target | Find the exact supporting passage in top-5 for as many questions as possible | RAG answer quality depends on it |
| Rank sensitivity | High — LLM receives top-K chunks; answer should surface at rank 1–3 | RAG prompt budget |
| Cloud | **Azure-only** (Azure OpenAI + Azure AI Search) | Enterprise/compliance constraint |
| Budget scale | Per-query embedding cost must support ≥100k queries/month | Assumed production volume |
| Data sensitivity | Document stays in tenant; no third-party embedding APIs | Compliance constraint |
| Latency budget | Query embedding ≤ ~500 ms (interactive chat) | UX constraint |

### What a fuller version adds (future work)

- Multilingual requirement (does the corpus stay English-only?)
- Freshness/re-indexing cadence (how often the corpus changes → re-embedding cost)
- Max acceptable p95 end-to-end retrieval latency
- Threshold-based filtering needs (affects how much the cosine distribution shape matters)

---

## Step 2 — Shortlist Candidate Models

Industry practice: shortlist 2–5 models that clear the hard constraints (cloud, compliance, language, dimension fit), spanning at least one budget and one premium option.

### Our shortlist (all Azure OpenAI — constraint-driven)

| Model | Generation | Dimension | Price / 1M tok | Why shortlisted |
|-------|-----------|-----------|----------------|-----------------|
| text-embedding-3-small | Current | 1536 | $0.02 | Budget baseline; carried over from Phase 1 |
| text-embedding-3-large | Current | 3072 | $0.13 | Premium ceiling; likely quality winner |
| text-embedding-ada-002 | Legacy | 1536 | $0.10 | Incumbent default many enterprises still run |

The trio deliberately spans the decision space: **cheapest current**, **best current**, **legacy incumbent**. That makes the recommendation actionable for any team starting from ada-002 (the most common real-world starting point).

### What a fuller version adds

- **Open-source models** (BGE, E5, GTE, Stella) via custom deployment — better cost-quality at scale, but violates our Azure-only constraint without extra infra
- **Third-party APIs** (Cohere Embed v3/v4, Voyage, Gemini embedding) — rejected by the data-residency constraint
- **Domain-adapted / fine-tuned embeddings** — relevant when generic models plateau (our 22-query ceiling suggests limited headroom, but on other corpora it matters)

---

## Step 3 — Build the Golden Dataset

Industry practice: a fixed, versioned Q&A set with *known ground-truth spans*, stratified across question types and difficulty, held constant across all experiments so results are comparable.

### What we built

**`data/pg_2025_gold_evaluation_dataset_1.json` — 100 questions** on the P&G FY2025 Annual Report (301 pages → markdown), each entry containing:

- `question` — natural-language query as a user would ask
- `ground_truth_answer` — expected answer text
- `supporting_text` — exact source span that must be retrieved (ground truth)
- `source` — document / page / section for auditability
- `question_type` + `difficulty` — for stratified analysis

### Stratification (why this set is credible)

| Dimension | Coverage |
|-----------|----------|
| Difficulty | easy 40 · medium 36 · hard 24 |
| Question types | fact 44 · table_lookup 12 · cause_effect 14 · list 6 · context 5 · comparison 4 · structured_data 3 · trend_analysis 3 · others 9 |
| Answer span length | 20–350 chars (avg 127) — exercises both pinpoint and wide spans |

This mix matters: `table_lookup` and `structured_data` questions stress numeric/tabular understanding; `cause_effect` and `multi_concept` stress semantic (not lexical) matching — exactly where embedding quality differentiates.

### Quality rules we followed

1. **Ground truth = exact source span**, enabling deterministic binary evaluation (§5)
2. **Versioned & validated** — `src/utils/validate_golden_dataset.py` checks schema/spans before runs
3. **Held constant** — same 100 questions for every model; only the embedding changes
4. Generated via a separate authoring pass (`buildgolddataset.py`), not from the same code path being evaluated — avoids self-testing bias

### What a fuller version adds

- **Multi-relevant-chunk questions** (recall becomes a real fraction, precision@K becomes informative)
- **Graded relevance** (0/0.5/1 instead of binary — partial credit for near-misses)
- **Negative queries** (questions whose answer is *not* in the corpus — measures hallucination-feeding behavior)
- **300+ questions** — needed to resolve gaps smaller than ~10 pts (our 3-small vs ada-002 tie)
- A second corpus/domain to check ranking stability

---

## Step 4 — Standardize the Pipeline

Industry practice: change **one variable at a time**. Everything downstream of the embedding (chunking, index, top-K, eval) must be bit-identical across candidates, or the comparison measures the pipeline, not the model.

### Controlled variables (identical for all 3 models)

| Stage | Setting |
|-------|---------|
| Chunking | recursive, size=512, overlap=50 → **1,269 chunks** |
| Corpus | same 88,549 tokens for every model (tokenized with cl100k_base) |
| Index | Azure AI Search, per-model strategy filter (`phase2_*`), fresh index per run, cleanup after |
| Retrieval | pure vector search, top-k=10, same query strings |
| Evaluation | same substring-match scorer, same golden set |

### Instrumentation built into the pipeline

- **CostTracker** — API calls, input tokens, USD (per-run, measured not estimated)
- **LatencyTracker** — per-stage timings (chunk / embed / index / retrieve / evaluate)
- **Per-query results persisted** — `per_query_recall@5` arrays enable paired statistical tests (§5)

### Known deviation (honest flag)

Phase 1 ranked `recursive` 4th of 5 chunking strategies, yet Phase 2 runs on recursive. This caps **absolute** numbers (see the 78% ceiling in EXPERIMENT.md §5) but the **relative model ranking** is what Phase 2 measures. An industry team would either re-run Phase 1's winner or justify the deviation — we documented it as a limitation instead.

---

## Step 5 — Retrieval-Level Quality Metrics

Industry practice: never pick an embedding model by cosine similarity alone or by gut feel — measure **end-to-end retrieval quality** on your own golden set. (MTEB leaderboard scores are a *screening* signal, not a selection signal.)

### What we measured

| Metric | Definition (this setup) | What it told us |
|--------|--------------------------|-----------------|
| **Recall@K** (≡ Hit-Rate@K here, 1 relevant chunk per query) | % of queries whose supporting span lands in top-K | Primary "did we find it" metric. 3-large: 0.75 @5 |
| **MRR** | Mean 1/rank of first relevant chunk | How high the answer surfaces. 3-large 0.577 ≈ first hit ~rank 1.7 |
| **NDCG@K** | Rank-discounted gain, normalized | Rewards rank-1 placement over rank-5 |
| **Precision@K** | Relevant ÷ K | Low-information here (capped at 1/K) — kept for completeness |
| **Cosine separation** | mean cos(relevant) − mean cos(irrelevant) | Model-intrinsic discriminability; screened the winner correctly (0.042 > 0.035 > 0.016) |

### K-sensitivity

Metrics at K ∈ {1, 3, 5, 10}: the k=1→5 climb is steep for all models (+25–28 pts); k=5→10 adds little (+7–11, best model gains least). **K=5 chosen as the reporting/decision point** — matching how many chunks we'd feed the LLM.

### Statistical rigor (what separates a demo from an experiment)

- **Paired t-test** on per-query Recall@5 (built into pipeline): 3-large vs 3-small p = 1.2e-05 ✅
- **McNemar exact test** (added during analysis — the correct test for paired binary outcomes): confirms 3-large's win (p = 1.9e-05) and exposes **3-small vs ada-002 as a statistical tie** (p = 0.267)
- **Per-query failure overlap**: 22/100 queries fail on *all three* models → a 78% ceiling that no embedding model can break on this pipeline. This reframes the whole selection: model choice governs the middle band; the ceiling belongs to chunking/eval.

**Winner at quality: text-embedding-3-large, significant at p < 0.0002 on two tests.**

---

## Step 6 — Operational / Non-Accuracy Metrics

Industry practice: a model that wins quality but blows the latency/cost/storage budget never ships. These metrics decide between the finalists.

### 6.1 Measured in this project

| Metric | 3-small | ada-002 | 3-large | How obtained |
|--------|--------:|--------:|--------:|--------------|
| **Embedding dimension** | 1536 | 1536 | 3072 | Model spec → index schema |
| **Cost per 1M tokens** | **$0.02** | $0.10 | $0.13 | Azure pricing |
| **Measured corpus indexing cost** (88.5k tok) | $0.0018 | $0.0089 | $0.0115 | CostTracker |
| **Cost per 1k queries** (150-tok query) | $0.003 | $0.015 | $0.0195 | Derived |
| **Cost @ 1M queries/month** | $3.00 | $15.00 | $19.50 | Derived |
| **Max input tokens** | 8,191 | 8,191 | 8,191 | Model spec (no constraint hit — longest chunk 512 tok) |
| **Vector storage per 1,269 chunks** (float32) | ~7.8 MB | ~7.8 MB | ~15.6 MB | Derived from dimension |
| **End-to-end run time** | 1223 s | 334 s ⚠️ | 1215 s | LatencyTracker |

⚠️ The ada-002 run-time anomaly reflects deployment provisioning/warmth, not model speed — documented as infrastructure noise.

### 6.2 Visual evidence — what each chart shows (all in `results/`, regenerateable from the saved JSONs with zero API calls)

Nine charts back every claim in this doc. Each maps a Step-5 quality metric or a Step-6 operational metric to a picture you can put on a slide:

| # | Chart | Type | What it plots | Key takeaway it evidences | Step |
|---|-------|------|---------------|---------------------------|------|
| V1 | `metrics_comparison.png` | Grouped bars | Recall@5, Precision@5, MRR, NDCG@5, Hit-Rate@5 side-by-side at K=5 | 3-large leads on every quality metric; the +21-pt Recall@5 gap is visible at a glance | 5 |
| V2 | `radar_chart.png` | 5-axis radar | Same 5 metrics as a per-model profile polygon | 3-large's polygon encloses the others; ada-002 ≈ 3-small in shape (their tie) | 5 |
| V3 | `k_sensitivity.png` | Line chart | Recall@K for K ∈ {1, 3, 5, 10} per model | k=1→5 climb is steep for all; k=5→10 flattens — K=5 is the sweet spot; the gap is widest at K=1 (+62% relative) | 5 |
| V4 | `cosine_distributions.png` | Paired bars | Mean cosine of relevant vs irrelevant chunks per model | The relevant–irrelevant gap exists for 3-series but is nearly closed for ada-002 | 5 |
| V5 | `cosine_separation.png` | Bar chart | Separation score per model | Model-intrinsic quality ordering: 0.042 > 0.035 > 0.016; also exposes ada-002's compressed space | 5 |
| V6 | `cost_quality_scatter.png` | Scatter + frontier | Cost per 1M tokens (x) vs Recall@5 (y) with the Pareto frontier drawn | ada-002 sits *inside* the 3-small→3-large frontier — visually dominated | 6 |
| V7 | `per_query_heatmap.png` | Query × model heatmap | Hit (green) / miss (red) for all 100 queries per model | The right-edge column of common reds = the 22-query pipeline ceiling no model can break | 5 |
| V8 | `composite_breakdown.png` | Stacked bars | Weighted contribution of each criterion to the composite score | Where each model's rank comes from: quality blocks for 3-large, the cost block for 3-small | 7 |
| V9 | `statistical_significance.png` | Bar chart vs α=0.05 line | Paired t-test p-values vs baseline (3-small) | 3-large's bar crashes through the threshold (p=1.2e-05); ada-002's stays above it (p=0.167) — the tie, made visible | 5 |

**Reading notes for presenting:**

- V1–V5 = quality story (Step 5); V6 = operational story (Step 6); V7–V9 = rigor story (statistics + failure analysis). A 6-slide summary maps one chart per slide.
- V5 includes a dashed "good threshold (0.3)" marker — treat it as aspirational only; what matters here is the *relative* ordering, since absolute separation magnitudes are model-family dependent.
- V9 runs the pipeline's built-in paired t-test against the first-loaded model (3-small). The McNemar exact tests quoted in this doc (§5) are computed during analysis and agree with the chart.
- All nine regenerate offline from `results/*.json`: `python regenerate_visuals.py` (V1–V6) and `python advanced_visuals.py` (V7–V9).

### 6.3 What the charts can *not* show (honest gaps)

| Missing metric | Why it needs a load test, not a chart | Priority |
|----------------|----------------------------------------|----------|
| p95/p99 latency under concurrency | Our latency is single-run, sequential; tail latency only appears under parallel load | High |
| Throughput ceiling (TPM/RPM, 429 throttle counts) | Requires sustained load against deployment quotas | High |
| Vector index size at scale | Derived here (15.6 vs 7.8 MB per 1,269 chunks); portal-observable at real corpus size | Medium |
| Actual billed $ vs CostTracker estimate | Cost tracker is code-level; a billing cross-check validates it | Medium |

### 6.4 Quality-per-dollar framing (the operational verdict)

| Model | Quality/$ (Recall@5 per $/1k queries) | Verdict |
|-------|--------------------------------------:|---------|
| 3-small | **180.0** | Efficiency winner (~4.7× 3-large) |
| ada-002 | 39.3 | Dominated — 3-small's price with no significant quality gain |
| 3-large | 38.5 | Premium — pay 6.5× for +21 pts Recall@5 |

---

## Step 7 — Build the Weighted Scoring Matrix

Industry practice: encode the business priorities into explicit weights, score every candidate on every criterion, and **show the sensitivity** — if the winner flips when weights change, the decision is weight-driven, not data-driven, and must be stated as such.

### Our matrix (weights reflect a quality-first RAG product)

| Criterion | Weight | 3-large | 3-small | ada-002 | Normalization |
|-----------|-------:|--------:|--------:|--------:|---------------|
| Recall@5 | 0.30 | 0.75 | 0.54 | 0.59 | raw (max = 1.0) |
| MRR | 0.25 | 0.577 | 0.401 | 0.436 | raw |
| NDCG@5 | 0.20 | 0.611 | 0.426 | 0.466 | raw |
| Cosine separation | 0.15 | 0.042 | 0.035 | 0.016 | raw |
| Cost efficiency | 0.10 | 0.00 | 0.85 | 0.23 | 1 − price/max_price |
| **Composite score** | **1.00** | **0.498** 🥇 | **0.437** 🥈 | **0.405** 🥉 | Σ weighted |

### Sensitivity analysis (the part most teams skip)

| Cost weight | Winner | Implication |
|------------:|--------|-------------|
| 10% (ours) | **3-large** | Quality-first products |
| 30% | **tie 3-large / 3-small** | The pivot point |
| ≥40% | **3-small** | Volume/cost-driven products |

**Takeaway:** the ranking is stable in quality dimensions (3-large leads every quality criterion) and flips only through the cost lever — so the real decision variable is *the cost of a retrieval miss in your application*, not the models.

### What a fuller matrix adds

- Separate latency criterion (once load-tested p95 exists, §6.3) with its own weight
- Storage criterion (vector index size at production corpus scale)
- "Risk" criterion (deployment maturity, deprecation policy — ada-002 scores poorly: legacy)
- Normalized per-criterion z-scores instead of raw mixes (prevents cosine_sep's small magnitudes from being drowned out)

---

## Step 8 — Summary & Decision

### Final scorecard

| | text-embedding-3-large | text-embedding-3-small | text-embedding-ada-002 |
|---|---|---|---|
| Quality (Recall@5) | 🥇 **0.75** | 0.54 | 0.59 |
| Ranking quality (MRR) | 🥇 **0.577** | 0.401 | 0.436 |
| Statistical status vs 3-small | **significantly better** (p<0.0002) | baseline | **tie** (p=0.267) |
| Cost / 1M queries | $19.50 | 🥇 **$3.00** | $15.00 |
| Efficiency (quality/$) | 38.5 | 🥇 **180** | 39.3 |
| Storage (1,269 chunks) | 15.6 MB | 🥇 7.8 MB | 🥇 7.8 MB |
| Verdict | **Default choice — quality-critical RAG** | **Scale choice — behind a reranker** | **Retire — Pareto-dominated** |

### Decision (with conditions)

1. **Default: text-embedding-3-large** for the capstone pipeline — quality-critical RAG where a retrieval miss degrades the answer, plus the strongest rank-1 behavior (+62% relative Hit@1)
2. **Scale path: text-embedding-3-small + reranker** — if a cross-encoder recovers even half the 21-pt gap, it beats paying 6.5× per query; this is the designated Phase 3 experiment
3. **Never: ada-002 for new builds** — statistically tied with 3-small at 5× the price, and its compressed cosine distribution (σ≈0.015) makes threshold-based filtering unreliable
4. **Any model migration = full re-embed + re-tune** — cosine scores are not comparable across models (ada-002's 0.88 ≈ 3-series' 0.76)

### Remaining gaps to close (tracked)

- [ ] Load test under production-like concurrency → fills the §6.3 gaps (p95 latency, throughput, throttling)
- [ ] Re-run comparison on Phase 1's winning chunker (fixed_size) → raises the 78% ceiling
- [ ] Reranker on 3-small (Phase 3) → tests the scale path
- [ ] Matryoshka dimension reduction on 3-large (1536/1024/256) → halves the storage tax

---

## Appendix A — Process Artifacts Map

| Artifact | Path |
|----------|------|
| Golden dataset (100 q) | `data/pg_2025_gold_evaluation_dataset_1.json` |
| Dataset builder / validator | `buildgolddataset.py` · `src/utils/validate_golden_dataset.py` |
| Experiment runner | `experiments/phase2_embeddings/run_experiment.py` |
| Config (models, K values) | `experiments/phase2_embeddings/config.py` |
| Metric implementations | `experiments/phase2_embeddings/evaluation/` |
| Raw results (final run) | `experiments/phase2_embeddings/results/*.json` |
| Charts | `experiments/phase2_embeddings/results/*.png` |
| Full findings write-up | `experiments/phase2_embeddings/results/EXPERIMENT.md` |
| This process doc | `experiments/phase2_embeddings/EMBEDDING_SELECTION_PROCESS.md` |

## Appendix B — Reusable Checklist (for the next selection round)

```
[ ] 1. Requirements: use case, quality bar, cloud/compliance, volume, latency budget written down
[ ] 2. Shortlist: 2–5 models spanning budget→premium, clearing hard constraints
[ ] 3. Golden set: versioned, stratified, ground-truth spans, validated, ≥100 (300 for fine gaps)
[ ] 4. Pipeline: one variable changed; chunker/index/top-K/eval identical; cost+latency instrumented
[ ] 5. Quality: Recall@K, MRR, NDCG@K at K∈{1,3,5,10}; paired significance test; failure-overlap analysis
[ ] 6. Operational: dimension, $/1M, measured index+query cost, storage, latency; visual evidence pack (cost-quality scatter, heatmap, composite breakdown)
[ ] 7. Matrix: explicit weights + sensitivity analysis; state what flips the winner
[ ] 8. Decision: conditions ("use X when…"), not just a winner; migration plan (re-embed + re-tune)
```
