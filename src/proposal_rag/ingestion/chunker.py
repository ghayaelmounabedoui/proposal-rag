import hashlib

from proposal_rag.models import Chunk, DocumentBlock


def make_id(value: str) -> str:
    """Créer un identifiant déterministe à partir d'un texte."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chunk_document(
    blocks: list[DocumentBlock],
    source: str,
) -> list[Chunk]:
    """Créer un chunk par paragraphe, accompagné des titres parents."""

    document_id = make_id(source)

    headings: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    paragraph_index = 0

    for block in blocks:
        if not block.text.strip():
            continue

        if block.is_heading:
            # Retirer les titres du même niveau ou d'un niveau inférieur.
            while headings and headings[-1][0] >= block.heading_level:
                headings.pop()

            headings.append((block.heading_level, block.text))
            continue

        heading_path = tuple(title for _, title in headings)

        # La position distingue deux paragraphes identiques.
        identity = (
            document_id,
            paragraph_index,
            heading_path,
            block.text,
        )

        chunks.append(
            Chunk(
                chunk_id=make_id(repr(identity)),
                document_id=document_id,
                source=source,
                content=block.text,
                heading_path=heading_path,
                page_start=block.page_start,
                page_end=block.page_end,
            )
        )

        paragraph_index += 1

    return chunks
