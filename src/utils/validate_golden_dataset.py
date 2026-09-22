"""
Validate the golden evaluation dataset for retrieval-testability.

Replicates the runtime evaluation contract exactly:
  1. supporting_text must be a verbatim (case-insensitive) substring of the source markdown,
     because retrieval_metrics.is_relevant() does `supporting_text.lower() in retrieved_text.lower()`.
  2. supporting_text must be fully contained in at least ONE chunk produced by the actual
     chunking pipeline (recursive_chunks, 512/50) — text straddling a chunk boundary can
     never be retrieved and silently fails every metric.
  3. No duplicate questions or duplicate supporting texts (identical text = identical
     retrieval results, which inflates apparent variety).

Usage (from project root):
    .venv/Scripts/python src/utils/validate_golden_dataset.py
Exit code 0 = all valid, 1 = problems found.
"""
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR, GOLDEN_DATASET_PATH
from src.utils.document_loader import load_all_documents
from src.chunking.recursive import recursive_chunks

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def find_closest_line(md_text: str, needle: str) -> str:
    """Return the markdown line most similar to the failed supporting_text."""
    lines = [ln for ln in md_text.splitlines() if ln.strip()]
    closest = difflib.get_close_matches(needle, lines, n=1, cutoff=0.4)
    return closest[0] if closest else ""


def validate(entries: list[dict], md_text: str, chunks: list) -> list[str]:
    problems = []
    md_lower = md_text.lower()
    chunk_texts_lower = [c.text.lower() for c in chunks]

    seen_questions: dict[str, str] = {}
    seen_supporting: dict[str, str] = {}

    for entry in entries:
        qid = entry.get("id", "<no-id>")
        st = entry.get("supporting_text", "")
        q = entry.get("question", "")

        if not st:
            problems.append(f"{qid}: empty supporting_text")
            continue

        # 1. Verbatim substring of the source markdown (runtime contract)
        if st.lower() not in md_lower:
            closest = find_closest_line(md_text, st)
            problems.append(
                f"{qid}: supporting_text NOT found verbatim in markdown.\n"
                f"    text: {st[:120]}...\n"
                f"    closest line: {closest[:160]}"
            )

        # 2. Fully contained in at least one chunk
        st_lower = st.lower()
        if not any(st_lower in ct for ct in chunk_texts_lower):
            problems.append(
                f"{qid}: supporting_text is NOT contained in any single chunk "
                f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}). It straddles a "
                f"chunk boundary and can never be retrieved. Shorten or shift the text.\n"
                f"    text: {st[:120]}..."
            )

        # 3. Duplicates
        if q.lower() in seen_questions:
            problems.append(f"{qid}: duplicate question of {seen_questions[q.lower()]}")
        else:
            seen_questions[q.lower()] = qid

        if st.lower() in seen_supporting:
            problems.append(
                f"{qid}: duplicate supporting_text of {seen_supporting[st.lower()]} "
                f"(identical text always yields identical retrieval results)"
            )
        else:
            seen_supporting[st.lower()] = qid

    return problems


def main():
    print("Loading golden dataset and building chunks...")
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"  {len(entries)} golden entries in {GOLDEN_DATASET_PATH.name}")

    md_files = sorted((DATA_DIR / "markdown").glob("*.md"))
    md_text = "\n".join(p.read_text(encoding="utf-8") for p in md_files)

    pages = load_all_documents()
    chunks = recursive_chunks(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(pages)} sections, {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    problems = validate(entries, md_text, chunks)

    if problems:
        print(f"FAILED: {len(problems)} problem(s) found\n")
        for p in problems:
            print(f"  - {p}\n")
        sys.exit(1)
    else:
        print(f"OK: all {len(entries)} entries are valid — verbatim in source, "
              f"chunk-contained, no duplicates.")


if __name__ == "__main__":
    main()
