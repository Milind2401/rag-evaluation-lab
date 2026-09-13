import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tiktoken
from src.chunking.base import Chunk


def fixed_size_chunks(
    pages: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split text into fixed-size token windows."""
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    chunk_idx = 0

    for page in pages:
        tokens = enc.encode(page["text"])
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(Chunk(
                id=f"fixed_{page['source']}_{page['page_number']}_{chunk_idx}",
                text=chunk_text,
                source=page["source"],
                page=page["page_number"],
                chunk_index=chunk_idx,
                metadata={"strategy": "fixed_size", "chunk_size": chunk_size, "overlap": chunk_overlap},
            ))
            chunk_idx += 1
            start += chunk_size - chunk_overlap
    return chunks


if __name__ == "__main__":
    from src.utils.document_loader import load_all_documents
    pages = load_all_documents()
    print(f"Loaded {len(pages)} pages\n")

    chunks = fixed_size_chunks(pages, chunk_size=512, chunk_overlap=50)
    print(f"Strategy: fixed_size | Chunk size: 512 | Overlap: 50")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:10]):
        print(f"--- Chunk {i+1} | Page {chunk.page} | Tokens: {len(chunk.text.split())} ---")
        print(chunk.text[:300])
        print()

    print(f"... ({len(chunks)} total chunks)")
