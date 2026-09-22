"""Generate cosine similarity distribution plots for Phase 2."""
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def generate_cosine_plots(results: list[dict], output_dir: Path):
    """Generate violin plots and histograms of cosine similarity distributions."""
    if not HAS_MPL:
        print("  Skipping cosine plots: matplotlib not installed")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [r["model"].replace("text-embedding-", "") for r in results]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # --- Violin Plot: Relevant vs Irrelevant per model ---
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    data_rel = []
    data_irrel = []
    labels = []

    for i, r in enumerate(results):
        dist = r.get("cosine_distributions", {})
        rel_mean = dist.get("relevant", {}).get("mean", 0)
        rel_std = dist.get("relevant", {}).get("std", 0)
        irrel_mean = dist.get("irrelevant", {}).get("mean", 0)
        irrel_std = dist.get("irrelevant", {}).get("std", 0)

        data_rel.append(rel_mean)
        data_irrel.append(irrel_mean)

    x = np.arange(len(results))
    width = 0.35

    bars1 = ax.bar(x - width / 2, data_rel, width, label="Relevant", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x + width / 2, data_irrel, width, label="Irrelevant", color="#F44336", alpha=0.8)

    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Phase 2: Cosine Similarity - Relevant vs Irrelevant Chunks")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = output_dir / "cosine_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Cosine Separation Bar Chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    separations = [r.get("cosine_distributions", {}).get("cosine_separation", 0) for r in results]
    bars = ax.bar(model_names, separations, color=colors[:len(results)], alpha=0.8)

    ax.set_ylabel("Cosine Separation Score")
    ax.set_title("Phase 2: Embedding Space Quality (Separation = Mean Relevant - Mean Irrelevant)")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Good threshold (0.3)")
    ax.legend()

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = output_dir / "cosine_separation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
