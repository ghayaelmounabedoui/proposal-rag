
from pathlib import Path

from proposal_rag.embeddings.local import LocalEmbedder
from proposal_rag.indexes.chroma_store import ChromaStore
from proposal_rag.ingestion.chunker import chunk_document
from proposal_rag.ingestion.loader import load_document


class DocumentIngestionService:
    def __init__(
        self,
        embedder: LocalEmbedder | None = None,
        store: ChromaStore | None = None,
    ):
        # Réutiliser les instances déjà chargées évite de recharger
        # le modèle d'embeddings à chaque upload.
        self.embedder = embedder or LocalEmbedder()
        self.store = store or ChromaStore()

    def index(self, path: Path, source: str) -> dict:
        blocks = load_document(path)
        chunks = chunk_document(blocks, source=source)

        if not chunks:
            raise ValueError("Aucun contenu à indexer.")

        embeddings = self.embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )

        self.store.upsert_chunks(chunks, embeddings)

        return {
            "source": source,
            "chunks_indexed": len(chunks),
            "total_chunks": self.store.count(),
        }
