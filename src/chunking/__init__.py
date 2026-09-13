from src.chunking.fixed_size import fixed_size_chunks
from src.chunking.recursive import recursive_chunks
from src.chunking.semantic import semantic_chunks
from src.chunking.markdown_aware import markdown_aware_chunks
from src.chunking.parent_child import parent_child_chunks

CHUNKING_FUNCTIONS = {
    "fixed_size": fixed_size_chunks,
    "recursive": recursive_chunks,
    "semantic": semantic_chunks,
    "markdown_aware": markdown_aware_chunks,
    "parent_child": parent_child_chunks,
}
