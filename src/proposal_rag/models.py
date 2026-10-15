from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentBlock:
    """Un titre ou un paragraphe extrait d'un document."""

    text: str
    heading_level: int = 0
    page_start: int | None = None
    page_end: int | None = None

    @property
    def is_heading(self) -> bool:
        return self.heading_level > 0


@dataclass(frozen=True)
class Chunk:
    """Un paragraphe complet accompagné de ses titres parents."""

    chunk_id: str
    document_id: str
    source: str
    content: str
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None

    @property
    def text(self) -> str:
        """Texte utilisé pour la recherche et la génération."""
        context = " > ".join(self.heading_path) or self.source
        return f"{context}\n\n{self.content}"
