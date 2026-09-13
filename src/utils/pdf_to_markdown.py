import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pymupdf4llm
from src.config import SOURCE_DOCS_DIR, DATA_DIR


def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path = None) -> Path:
    """Convert a single PDF to markdown using pymupdf4llm."""
    if output_dir is None:
        output_dir = DATA_DIR / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"{pdf_path.stem}.md"
    if md_path.exists():
        print(f"  Already exists: {md_path}")
        return md_path

    print(f"  Converting {pdf_path.name} to markdown...")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    md_path.write_text(md_text, encoding="utf-8")
    print(f"  Saved: {md_path} ({len(md_text)} chars)")
    return md_path


def convert_all_pdfs() -> list[Path]:
    """Convert all PDFs in source_documents to markdown."""
    pdfs = list(SOURCE_DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {SOURCE_DOCS_DIR}")
        return []

    print(f"Found {len(pdfs)} PDF(s)")
    md_paths = []
    for pdf in pdfs:
        md_path = convert_pdf_to_markdown(pdf)
        md_paths.append(md_path)
    return md_paths


if __name__ == "__main__":
    md_paths = convert_all_pdfs()
    for p in md_paths:
        print(f"\n--- Preview: {p.name} (first 500 chars) ---")
        print(p.read_text(encoding="utf-8")[:500])
