import json
from pathlib import Path

import chromadb

from proposal_rag.models import Chunk


class ChromaStore:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        database_path = project_root / "var" / "chroma"

        self.client = chromadb.PersistentClient(
            path=str(database_path)
        )

        # Collection dédiée au modèle E5 small, 384 dimensions.
        self.collection = self.client.get_or_create_collection(
            name="company-e5-small-v1",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Enregistrer les chunks ou actualiser les IDs déjà présents."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chaque chunk doit avoir exactement un embedding."
            )

        if not chunks:
            return

        if any(len(vector) != 384 for vector in embeddings):
            raise ValueError("Les vecteurs doivent avoir 384 dimensions.")

        metadatas = []

        for chunk in chunks:
            metadata = {
                "document_id": chunk.document_id,
                "source": chunk.source,
                "content": chunk.content,
                "heading_path": json.dumps(
                    chunk.heading_path,
                    ensure_ascii=False,
                ),
            }

            # Ne pas enregistrer None comme valeur de métadonnée.
            if chunk.page_start is not None:
                metadata["page_start"] = chunk.page_start

            if chunk.page_end is not None:
                metadata["page_end"] = chunk.page_end

            metadatas.append(metadata)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()
    def query(
    self,
    query_embedding: list[float],
    top_k: int = 5,
) -> dict:
        if top_k <= 0:
            raise ValueError("top_k doit être supérieur à 0.")

        if self.collection.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    def get_all(self) -> dict:
        return self.collection.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

    
    def list_documents(self) -> list[dict]:
        data = self.collection.get(
            include=["metadatas"]
        )

        documents = {}

        for metadata in data["metadatas"]:
            if not metadata:
                continue

            source = metadata.get("source")

            if not source:
                continue

            if source not in documents:
                documents[source] = {
                    "filename": source,
                    "chunks": 0,
                }

            documents[source]["chunks"] += 1

        return sorted(
            documents.values(),
            key=lambda doc: doc["filename"],
        )

    def delete_document(self, source: str) -> int:
        data = self.collection.get(
            where={"source": source},
            include=["metadatas"],
        )

        ids = data["ids"]

        if not ids:
            return 0

        self.collection.delete(ids=ids)

        return len(ids)
