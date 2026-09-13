import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.chunking.base import Chunk


def recursive_chunks(
    pages: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split text recursively by separators."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    chunk_idx = 0

    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append(Chunk(
                id=f"rec_{page['source']}_{page['page_number']}_{chunk_idx}",
                text=split,
                source=page["source"],
                page=page["page_number"],
                chunk_index=chunk_idx,
                metadata={"strategy": "recursive", "chunk_size": chunk_size, "overlap": chunk_overlap},
            ))
            chunk_idx += 1
    return chunks


if __name__ == "__main__":
    from src.utils.document_loader import load_all_documents
    pages = load_all_documents()
    print(f"Loaded {len(pages)} pages\n")

    chunks = recursive_chunks(pages, chunk_size=512, chunk_overlap=50)
    print(f"Strategy: recursive | Chunk size: 512 | Overlap: 50")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:10]):
        print(f"--- Chunk {i+1} | Page {chunk.page} | Tokens: {len(chunk.text.split())} ---")
        print(chunk.text[:300])
        print()

    print(f"... ({len(chunks)} total chunks)")
