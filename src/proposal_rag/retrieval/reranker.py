import os

from sentence_transformers import CrossEncoder

from proposal_rag.retrieval.models import RetrievalResult

# Modèle configurable : ms-marco est entraîné sur de l'anglais.
# Pour des documents surtout français, définir RERANKER_MODEL
# (ex. un cross-encoder multilingue) puis réindexer n'est pas nécessaire.
DEFAULT_RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        query = query.strip()

        if not query:
            raise ValueError("La question ne peut pas être vide.")

        if top_k <= 0:
            raise ValueError("top_k doit être supérieur à 0.")

        if not results:
            return []

        pairs = [
            (query, result.text)
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = [
            RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),
                metadata=result.metadata,
            )
            for result, score in zip(
                results,
                scores,
                strict=False,
            )
        ]

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked[:top_k]
