from proposal_rag.retrieval import hybrid
from proposal_rag.retrieval.bm25 import BM25Retriever, tokenize
from proposal_rag.retrieval.hybrid import HybridRetriever
from proposal_rag.retrieval.models import RetrievalResult


def result(chunk_id, text="texte"):
    return RetrievalResult(chunk_id=chunk_id, text=text, score=0.0)


class FakeDense:
    def __init__(self, results):
        self.results = results
        self.store = object()

    def search(self, query, top_k):
        return self.results[:top_k]


def test_tokenize_lowercases_and_keeps_accents():
    assert tokenize("Sécurité AES-256!") == ["sécurité", "aes", "256"]


def test_bm25_finds_keyword_and_ignores_unrelated():
    bm25 = BM25Retriever(
        chunk_ids=["a", "b", "c"],
        texts=[
            "chiffrement AES-256 des données",
            "sauvegardes quotidiennes",
            "authentification multifacteur",
        ],
        metadatas=[{}, {}, {}],
    )

    hits = bm25.search("sauvegardes", top_k=3)

    assert [h.chunk_id for h in hits] == ["b"]
    assert bm25.search("inexistant", top_k=3) == []


def test_bm25_empty_corpus():
    bm25 = BM25Retriever([], [], [])
    assert bm25.search("anything") == []


def test_rrf_prefers_chunks_found_by_both_retrievers():
    dense = FakeDense([result("x"), result("shared"), result("y")])
    bm25 = BM25Retriever(
        chunk_ids=["shared", "z", "w", "v"],
        texts=["shared requirement text", "other", "autre", "encore"],
        metadatas=[{}, {}, {}, {}],
    )

    retriever = HybridRetriever(dense, bm25)
    hits = retriever.search("shared requirement", top_k=3)

    assert hits[0].chunk_id == "shared"


def test_refresh_rebuilds_bm25_from_store(monkeypatch):
    dense = FakeDense([])
    old = BM25Retriever(["a"], ["ancien texte"], [{}])
    # BM25 donne un score nul aux mots présents dans la moitié du corpus.
    new = BM25Retriever(
        ["b", "c", "d", "e"],
        ["nouveau document", "autre sujet", "encore un", "dernier"],
        [{}, {}, {}, {}],
    )

    monkeypatch.setattr(
        hybrid, "build_bm25_from_chroma", lambda store: new
    )

    retriever = HybridRetriever(dense, old)
    assert retriever.search("nouveau", top_k=3) == []

    retriever.refresh()

    assert [h.chunk_id for h in retriever.search("nouveau", top_k=3)] == ["b"]
