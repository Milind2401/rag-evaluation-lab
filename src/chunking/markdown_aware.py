import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
from src.chunking.base import Chunk


def markdown_aware_chunks(
    pages: list[dict],
    max_chunk_size: int = 1024,
) -> list[Chunk]:
    """Split by markdown headings, keeping tables intact."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    chunks = []
    chunk_idx = 0

    for page in pages:
        text = page["text"]
        headings = list(heading_pattern.finditer(text))

        if not headings:
            chunks.append(Chunk(
                id=f"md_{page['source']}_{page['page_number']}_{chunk_idx}",
                text=text,
                source=page["source"],
                page=page["page_number"],
                section="full_page",
                chunk_index=chunk_idx,
                metadata={"strategy": "markdown_aware"},
            ))
            chunk_idx += 1
            continue

        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section_text = text[start:end].strip()
            heading_text = match.group(0)

            if len(section_text) > max_chunk_size:
                sub_chunks = _split_large_section(section_text, max_chunk_size)
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        id=f"md_{page['source']}_{page['page_number']}_{chunk_idx}",
                        text=sc,
                        source=page["source"],
                        page=page["page_number"],
                        section=heading_text,
                        chunk_index=chunk_idx,
                        metadata={"strategy": "markdown_aware"},
                    ))
                    chunk_idx += 1
            else:
                chunks.append(Chunk(
                    id=f"md_{page['source']}_{page['page_number']}_{chunk_idx}",
                    text=section_text,
                    source=page["source"],
                    page=page["page_number"],
                    section=heading_text,
                    chunk_index=chunk_idx,
                    metadata={"strategy": "markdown_aware"},
                ))
                chunk_idx += 1
    return chunks


def _split_large_section(text: str, max_size: int) -> list[str]:
    """Split a large section by paragraphs."""
    paragraphs = text.split("\n\n")
    result = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_size and current:
            result.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        result.append(current.strip())
    return result


if __name__ == "__main__":
    from src.utils.document_loader import load_all_documents
    pages = load_all_documents()
    print(f"Loaded {len(pages)} pages\n")

    chunks = markdown_aware_chunks(pages, max_chunk_size=1024)
    print(f"Strategy: markdown_aware | Max chunk size: 1024")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:10]):
        print(f"--- Chunk {i+1} | Page {chunk.page} | Section: {chunk.section} | Tokens: {len(chunk.text.split())} ---")
        print(chunk.text[:300])
        print()

    print(f"... ({len(chunks)} total chunks)")
