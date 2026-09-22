"""
Run a single embedding model for testing/debugging.
Usage: python run_text_embedding_small.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import EMBEDDING_MODELS, RESULTS_DIR
from run_experiment import run_single_model, save_result


def main():
    model_name = "text-embedding-3-small"
    model_config = EMBEDDING_MODELS[model_name]

    result = run_single_model(model_name, model_config)
    save_result(result, RESULTS_DIR)


if __name__ == "__main__":
    main()
