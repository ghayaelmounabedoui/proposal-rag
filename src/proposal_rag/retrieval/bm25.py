import re

from rank_bm25 import BM25Okapi

from proposal_rag.indexes.chroma_store import ChromaStore
from proposal_rag.retrieval.models import RetrievalResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        if not (
            len(chunk_ids)
            == len(texts)
            == len(metadatas)
        ):
            raise ValueError(
                "chunk_ids, texts et metadatas "
                "doivent avoir la même taille."
            )

        self.chunk_ids = chunk_ids
        self.texts = texts
        self.metadatas = metadatas

        self.tokenized_corpus = [
            tokenize(text)
            for text in texts
        ]

        self.index = (
            BM25Okapi(self.tokenized_corpus)
            if self.tokenized_corpus
            else None
        )

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

        if not self.texts or self.index is None:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            (
                index
                for index, score in enumerate(scores)
                if score > 0
            ),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                chunk_id=self.chunk_ids[index],
                text=self.texts[index],
                score=float(scores[index]),
                metadata=self.metadatas[index] or {},
            )
            for index in ranked_indices
        ]


def build_bm25_from_chroma(
    store: ChromaStore | None = None,
) -> BM25Retriever:
    store = store or ChromaStore()

    data = store.get_all()

    return BM25Retriever(
        chunk_ids=data.get("ids", []),
        texts=data.get("documents", []),
        metadatas=data.get("metadatas", []),
    )
