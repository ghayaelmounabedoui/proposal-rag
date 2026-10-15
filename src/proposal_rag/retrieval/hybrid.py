from collections import defaultdict

from proposal_rag.retrieval.bm25 import (
    BM25Retriever,
    build_bm25_from_chroma,
)
from proposal_rag.retrieval.dense import DenseRetriever
from proposal_rag.retrieval.models import RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k doit être supérieur à 0.")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def refresh(self) -> None:
        """Reconstruire BM25 depuis ChromaDB.

        À appeler après l'ajout ou la suppression d'un document, sinon
        la recherche par mots-clés ne voit pas les changements.
        """
        self.bm25_retriever = build_bm25_from_chroma(
            store=self.dense_retriever.store,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[RetrievalResult]:
        query = query.strip()

        if not query:
            raise ValueError("La question ne peut pas être vide.")

        if top_k <= 0:
            raise ValueError("top_k doit être supérieur à 0.")

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k doit être supérieur à 0."
            )

        candidate_k = max(candidate_k, top_k)

        dense_results = self.dense_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        rrf_scores: dict[str, float] = defaultdict(float)

        results_by_id: dict[str, RetrievalResult] = {}

        self._add_ranking(
            ranking=dense_results,
            scores=rrf_scores,
            results_by_id=results_by_id,
        )

        self._add_ranking(
            ranking=bm25_results,
            scores=rrf_scores,
            results_by_id=results_by_id,
        )

        ranked_ids = sorted(
            rrf_scores.keys(),
            key=lambda chunk_id: rrf_scores[chunk_id],
            reverse=True,
        )

        return [
            RetrievalResult(
                chunk_id=results_by_id[chunk_id].chunk_id,
                text=results_by_id[chunk_id].text,
                score=rrf_scores[chunk_id],
                metadata=results_by_id[chunk_id].metadata,
            )
            for chunk_id in ranked_ids[:top_k]
        ]

    def _add_ranking(
        self,
        ranking: list[RetrievalResult],
        scores: dict[str, float],
        results_by_id: dict[str, RetrievalResult],
    ) -> None:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] += (
                1.0 / (self.rrf_k + rank)
            )

            results_by_id.setdefault(
                result.chunk_id,
                result,
            )


def build_hybrid_retriever() -> HybridRetriever:
    dense_retriever = DenseRetriever()

    bm25_retriever = build_bm25_from_chroma(
        store=dense_retriever.store,
    )

    return HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )