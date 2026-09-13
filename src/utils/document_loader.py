import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pypdf import PdfReader
from src.config import SOURCE_DOCS_DIR, DATA_DIR


def load_pdf(file_path: Path) -> list[dict]:
    """Load PDF and return list of page dicts with page number and text."""
    reader = PdfReader(str(file_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page_number": i + 1,
                "text": text.strip(),
                "source": file_path.name,
            })
    return pages


def load_markdown(file_path: Path) -> list[dict]:
    """Load markdown file and split into page-like chunks by headings."""
    import re
    text = file_path.read_text(encoding="utf-8")
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headings = list(heading_pattern.finditer(text))

    pages = []
    if not headings:
        pages.append({
            "page_number": 1,
            "text": text.strip(),
            "source": file_path.name,
        })
        return pages

    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            pages.append({
                "page_number": i + 1,
                "text": section_text,
                "source": file_path.name,
            })
    return pages


def load_all_documents(use_markdown: bool = True) -> list[dict]:
    """Load all documents. Prefers markdown if available."""
    if use_markdown:
        md_dir = DATA_DIR / "markdown"
        md_files = list(md_dir.glob("*.md"))
        if md_files:
            all_pages = []
            for md in md_files:
                all_pages.extend(load_markdown(md))
            return all_pages

    all_pages = []
    for pdf_path in SOURCE_DOCS_DIR.glob("*.pdf"):
        all_pages.extend(load_pdf(pdf_path))
    return all_pages


def load_golden_dataset(path: Path = None) -> list[dict]:
    """Load the golden evaluation dataset."""
    import json
    from src.config import GOLDEN_DATASET_PATH
    target = path or GOLDEN_DATASET_PATH
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)
