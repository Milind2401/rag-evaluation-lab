from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    section: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
