"""Generate cost vs quality scatter plot for Phase 2."""
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def generate_cost_quality_plot(results: list[dict], comparison: dict, output_dir: Path):
    """Generate scatter plot of cost vs Recall@5 with model labels."""
    if not HAS_MPL:
        print("  Skipping cost-quality plot: matplotlib not installed")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, r in enumerate(results):
        cost = r["cost_per_million_tokens"]
        recall = r["retrieval_metrics"].get("recall@5", 0)
        label = r["model"].replace("text-embedding-", "")

        ax.scatter(cost, recall, s=200, c=colors[i % len(colors)], zorder=5, edgecolors="black")
        ax.annotate(f"  {label}", (cost, recall), fontsize=10, fontweight="bold")

    ax.set_xlabel("Cost per 1M Tokens ($)")
    ax.set_ylabel("Recall@5")
    ax.set_title("Phase 2: Cost vs Quality Tradeoff")
    ax.grid(alpha=0.3)

    # Draw Pareto frontier
    cost_recall = [(r["cost_per_million_tokens"], r["retrieval_metrics"].get("recall@5", 0)) for r in results]
    cost_recall.sort(key=lambda x: x[0])
    pareto_x = [c[0] for c in cost_recall]
    pareto_y = [c[1] for c in cost_recall]
    ax.plot(pareto_x, pareto_y, "--", color="gray", alpha=0.5, label="Cost-quality frontier")
    ax.legend()

    plt.tight_layout()
    path = output_dir / "cost_quality_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
