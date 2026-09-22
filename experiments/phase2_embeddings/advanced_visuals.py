"""
Generate advanced Phase 2 visuals from saved JSON results (no API calls needed).

Produces:
  1. per_query_heatmap.png       - query x model heatmap of per-query recall@5
  2. composite_breakdown.png     - stacked bar of weighted composite-score contributions
  3. statistical_significance.png- paired t-test p-values vs the 0.05 threshold

Usage (from project root):
    .venv/Scripts/python experiments/phase2_embeddings/advanced_visuals.py
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR

MODEL_FILES = ["3_small.json", "3_large.json", "ada_002.json"]
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]

# Composite weights from evaluation/comparison.py
W_RECALL, W_MRR, W_NDCG, W_SEP, W_COST = 0.30, 0.25, 0.20, 0.15, 0.10


def load_results():
    results = []
    for name in MODEL_FILES:
        path = RESULTS_DIR / name
        if path.exists():
            with open(path) as f:
                results.append(json.load(f))
    return results


def generate_per_query_heatmap(results, output_dir):
    """Query x model heatmap of per-query recall@5 (0 = miss, 1 = hit)."""
    model_names = [r["model"].replace("text-embedding-", "") for r in results]
    matrix = np.array([r["per_query_recall@5"] for r in results])  # (models, queries)

    fig, ax = plt.subplots(figsize=(16, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"Q{i+1}" for i in range(matrix.shape[1])], fontsize=6)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Golden dataset question")
    ax.set_title("Phase 2: Per-Query Recall@5 — Which Questions Each Model Misses\n(green = answer in top-5, red = miss)")

    # Overlay hit/miss markers
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "H" if matrix[i, j] else "M",
                    ha="center", va="center", fontsize=5,
                    color="black", alpha=0.6)

    # Summary row: total hits per model on the right side
    for i, r in enumerate(results):
        hits = int(sum(r["per_query_recall@5"]))
        ax.annotate(f"{hits}/{matrix.shape[1]}", xy=(1.02, (matrix.shape[0] - 1 - i) / matrix.shape[0]),
                    xycoords="axes fraction", fontsize=10, fontweight="bold", va="center")

    fig.colorbar(im, ax=ax, label="Recall@5", shrink=0.8)
    plt.tight_layout()
    path = output_dir / "per_query_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_composite_breakdown(results, output_dir):
    """Stacked bar showing weighted contributions to each model's composite score."""
    max_cost = max(r["cost_per_million_tokens"] for r in results)

    components = {c: [] for c in ["Recall@5 (30%)", "MRR (25%)", "NDCG@5 (20%)",
                                  "Cosine Separation (15%)", "Cost Efficiency (10%)"]}
    model_names = []

    for r in results:
        m = r["retrieval_metrics"]
        cost_eff = 1.0 - (r["cost_per_million_tokens"] / max_cost) if max_cost > 0 else 1.0
        sep = r["cosine_distributions"]["cosine_separation"]

        model_names.append(r["model"].replace("text-embedding-", ""))
        components["Recall@5 (30%)"].append(W_RECALL * m["recall@5"])
        components["MRR (25%)"].append(W_MRR * m["mrr"])
        components["NDCG@5 (20%)"].append(W_NDCG * m["ndcg@5"])
        components["Cosine Separation (15%)"].append(W_SEP * sep)
        components["Cost Efficiency (10%)"].append(W_COST * cost_eff)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    bottom = np.zeros(len(model_names))
    comp_colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#607D8B"]

    for (label, values), color in zip(components.items(), comp_colors):
        vals = np.array(values)
        ax.bar(x, vals, 0.55, bottom=bottom, label=label, color=color, alpha=0.85)
        # Label segments large enough to read
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.02:
                ax.text(xi, b + v / 2, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals

    # Total on top of each stack
    for xi, total in enumerate(bottom):
        ax.text(xi, total + 0.008, f"Total: {total:.4f}", ha="center",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Composite Score")
    ax.set_title("Phase 2: Composite Score Breakdown\n(0.30·R@5 + 0.25·MRR + 0.20·NDCG@5 + 0.15·Separation + 0.10·CostEff)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(bottom) * 1.15)

    plt.tight_layout()
    path = output_dir / "composite_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_significance_chart(results, output_dir):
    """Paired t-test p-values for each model vs baseline, against alpha = 0.05."""
    baseline = results[0]
    baseline_queries = baseline.get("per_query_recall@5", [])
    baseline_name = baseline["model"].replace("text-embedding-", "")

    pairs = []  # (label, p_value, significant, mean_diff)
    for r in results[1:]:
        model_queries = r.get("per_query_recall@5", [])
        if len(model_queries) != len(baseline_queries) or len(baseline_queries) < 2:
            continue
        t_stat, p_value = stats.ttest_rel(model_queries, baseline_queries)
        pairs.append((
            r["model"].replace("text-embedding-", "") + f"\nvs {baseline_name}",
            p_value,
            p_value < 0.05,
            float(np.mean(model_queries) - np.mean(baseline_queries)),
        ))

    if not pairs:
        print("  Skipping significance chart: no comparable model pairs")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [p[0] for p in pairs]
    p_values = [p[1] for p in pairs]
    sig_flags = [p[2] for p in pairs]
    diffs = [p[3] for p in pairs]
    bar_colors = ["#F44336" if s else "#BDBDBD" for s in sig_flags]

    bars = ax.bar(labels, p_values, color=bar_colors, alpha=0.85, edgecolor="black")
    ax.axhline(y=0.05, color="red", linestyle="--", linewidth=2, label="Significance threshold (α = 0.05)")

    for bar, p, s, d in zip(bars, p_values, sig_flags, diffs):
        y = bar.get_height()
        verdict = "SIGNIFICANT" if s else "not significant"
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.015,
                f"p = {p:.4f}\n({verdict})\nΔrecall = {d:+.2f}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if s else "normal")

    ax.set_ylabel("p-value")
    ax.set_title(f"Phase 2: Statistical Significance of Model Differences\n(paired t-test on per-query recall@5, {len(baseline_queries)} queries)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "statistical_significance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    if not HAS_MPL:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    results = load_results()
    if not results:
        print("No result files found. Run run_experiment.py first.")
        return

    print("Generating advanced visuals from saved results...")
    generate_per_query_heatmap(results, RESULTS_DIR)
    generate_composite_breakdown(results, RESULTS_DIR)
    generate_significance_chart(results, RESULTS_DIR)
    print(f"Done. PNGs written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
