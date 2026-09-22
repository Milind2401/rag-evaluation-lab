"""Generate metric comparison charts for Phase 2 embedding evaluation."""
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from pathlib import Path


def generate_metric_charts(results: list[dict], output_dir: Path):
    """Generate grouped bar chart and radar chart comparing all models."""
    if not HAS_MPL:
        print("  Skipping metric charts: matplotlib not installed")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [r["model"].replace("text-embedding-", "") for r in results]
    metrics_keys = ["recall@5", "precision@5", "mrr", "ndcg@5", "hit_rate@5"]
    metrics_labels = ["Recall@5", "Precision@5", "MRR", "NDCG@5", "Hit Rate@5 (%)"]

    # --- Grouped Bar Chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_keys))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, r in enumerate(results):
        values = [r["retrieval_metrics"].get(m, 0) for m in metrics_keys]
        # Normalize hit_rate to 0-1 scale for comparison
        values[4] = values[4] / 100.0
        ax.bar(x + i * width, values, width, label=model_names[i], color=colors[i % len(colors)])

    ax.set_ylabel("Score")
    ax.set_title("Phase 2: Embedding Model Comparison (K=5)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for i, r in enumerate(results):
        values = [r["retrieval_metrics"].get(m, 0) for m in metrics_keys]
        values[4] = values[4] / 100.0
        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Radar Chart ---
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics_keys), endpoint=False).tolist()
    angles += angles[:1]

    for i, r in enumerate(results):
        values = [r["retrieval_metrics"].get(m, 0) for m in metrics_keys]
        values[4] = values[4] / 100.0
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_names[i], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase 2: Multi-Metric Radar Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    path = output_dir / "radar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- K-Sensitivity Line Chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    k_values = [1, 3, 5, 10]

    for i, r in enumerate(results):
        recalls = [r["retrieval_metrics"].get(f"recall@{k}", 0) for k in k_values]
        ax.plot(k_values, recalls, "o-", linewidth=2, markersize=8,
                label=model_names[i], color=colors[i % len(colors)])

    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Phase 2: Recall vs K by Embedding Model")
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid(alpha=0.3)

    for i, r in enumerate(results):
        recalls = [r["retrieval_metrics"].get(f"recall@{k}", 0) for k in k_values]
        for k, v in zip(k_values, recalls):
            ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7)

    plt.tight_layout()
    path = output_dir / "k_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
