from proposal_rag.embeddings.local import LocalEmbedder
from proposal_rag.indexes.chroma_store import ChromaStore
from proposal_rag.retrieval.models import RetrievalResult


class DenseRetriever:
    def __init__(
        self,
        embedder: LocalEmbedder | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        self.embedder = embedder or LocalEmbedder()
        self.store = store or ChromaStore()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        query = query.strip()

        if not query:
            raise ValueError("La question ne peut pas être vide.")

        if top_k <= 0:
            raise ValueError("top_k doit être supérieur à 0.")

        if self.store.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        results = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievalResult] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            retrieved.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(distance),
                    metadata=metadata or {},
                )
            )

        return retrieved
