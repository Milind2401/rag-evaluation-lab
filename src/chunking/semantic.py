import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from src.chunking.base import Chunk
from src.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT


def semantic_chunks(
    pages: list[dict],
    chunk_size: int = 512,
    breakpoint_threshold: float = 75.0,
) -> list[Chunk]:
    """Split text based on semantic similarity between sentences."""
    embeddings = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_amount=breakpoint_threshold,
    )
    chunks = []
    chunk_idx = 0

    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append(Chunk(
                id=f"sem_{page['source']}_{page['page_number']}_{chunk_idx}",
                text=split,
                source=page["source"],
                page=page["page_number"],
                chunk_index=chunk_idx,
                metadata={"strategy": "semantic", "breakpoint_threshold": breakpoint_threshold},
            ))
            chunk_idx += 1
    return chunks


if __name__ == "__main__":
    from src.utils.document_loader import load_all_documents
    pages = load_all_documents()
    print(f"Loaded {len(pages)} pages\n")

    chunks = semantic_chunks(pages, chunk_size=512, breakpoint_threshold=75.0)
    print(f"Strategy: semantic | Breakpoint threshold: 75.0")
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:10]):
        print(f"--- Chunk {i+1} | Page {chunk.page} | Tokens: {len(chunk.text.split())} ---")
        print(chunk.text[:300])
        print()

    print(f"... ({len(chunks)} total chunks)")
