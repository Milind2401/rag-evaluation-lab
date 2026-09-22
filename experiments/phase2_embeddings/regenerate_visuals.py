"""
Regenerate Phase 2 visuals from saved JSON results (no API calls needed).

Usage (from project root):
    .venv/Scripts/python experiments/phase2_embeddings/regenerate_visuals.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR
from visualization.metric_charts import generate_metric_charts
from visualization.cosine_distributions import generate_cosine_plots
from visualization.cost_quality_scatter import generate_cost_quality_plot

MODEL_FILES = ["3_small.json", "3_large.json", "ada_002.json"]


def main():
    results = []
    for name in MODEL_FILES:
        path = RESULTS_DIR / name
        if not path.exists():
            print(f"  Skipping missing result file: {path}")
            continue
        with open(path) as f:
            results.append(json.load(f))

    if not results:
        print("No result files found. Run run_experiment.py first.")
        return

    comparison_path = RESULTS_DIR / "comparison_summary.json"
    comparison = {}
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)

    print("Generating visuals from saved results...")
    generate_metric_charts(results, RESULTS_DIR)
    generate_cosine_plots(results, RESULTS_DIR)
    generate_cost_quality_plot(results, comparison, RESULTS_DIR)
    print(f"Done. PNGs written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
