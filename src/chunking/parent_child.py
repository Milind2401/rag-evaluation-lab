import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.chunking.base import Chunk


def parent_child_chunks(
    pages: list[dict],
    parent_size: int = 1024,
    child_size: int = 256,
    child_overlap: int = 30,
) -> list[Chunk]:
    """Create small retrieval chunks linked to larger parent context chunks."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=0,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
    )
    chunks = []
    chunk_idx = 0

    for page in pages:
        parent_splits = parent_splitter.split_text(page["text"])
        for parent_idx, parent_text in enumerate(parent_splits):
            child_splits = child_splitter.split_text(parent_text)
            for child_text in child_splits:
                chunks.append(Chunk(
                    id=f"pc_{page['source']}_{page['page_number']}_{chunk_idx}",
                    text=child_text,
                    source=page["source"],
                    page=page["page_number"],
                    chunk_index=chunk_idx,
                    metadata={
                        "strategy": "parent_child",
                        "parent_text": parent_text,
                        "parent_index": parent_idx,
                    },
                ))
                chunk_idx += 1
    return chunks


if __name__ == "__main__":
    from src.utils.document_loader import load_all_documents
    pages = load_all_documents()
    print(f"Loaded {len(pages)} pages\n")

    chunks = parent_child_chunks(pages, parent_size=1024, child_size=256, child_overlap=30)
    print(f"Strategy: parent_child | Parent: 1024 | Child: 256 | Overlap: 30")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:10]):
        print(f"--- Chunk {i+1} | Page {chunk.page} | Tokens: {len(chunk.text.split())} ---")
        print(f"CHILD: {chunk.text[:200]}")
        parent_preview = chunk.metadata.get("parent_text", "")[:150]
        print(f"PARENT: {parent_preview}...")
        print()

    print(f"... ({len(chunks)} total chunks)")
