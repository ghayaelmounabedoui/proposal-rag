from proposal_rag.retrieval.hybrid import (
    HybridRetriever,
    build_hybrid_retriever,
)
from proposal_rag.retrieval.models import RetrievalResult
from proposal_rag.retrieval.reranker import Reranker


class RetrievalPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.hybrid_retriever = (
            hybrid_retriever
            or build_hybrid_retriever()
        )

        self.reranker = reranker or Reranker()

    def refresh(self) -> None:
        """Prendre en compte les documents ajoutés ou supprimés."""
        self.hybrid_retriever.refresh()

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

        if candidate_k < top_k:
            candidate_k = top_k

        candidates = self.hybrid_retriever.search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )
