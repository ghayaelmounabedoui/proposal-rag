from proposal_rag.embeddings.local import LocalEmbedder


def make_embedder(monkeypatch):
    """Embedder sans modèle : on enregistre seulement le préfixe utilisé."""
    embedder = object.__new__(LocalEmbedder)
    used = []

    def fake_encode(texts, prefix):
        used.append(prefix)
        return [[0.0] * 384 for _ in texts]

    monkeypatch.setattr(embedder, "_encode", fake_encode)
    return embedder, used


def test_documents_use_passage_prefix(monkeypatch):
    embedder, used = make_embedder(monkeypatch)

    embedder.embed_documents(["un paragraphe"])

    assert used == ["passage"]


def test_queries_use_query_prefix(monkeypatch):
    embedder, used = make_embedder(monkeypatch)

    embedder.embed_query("  une question ?  ")

    assert used == ["query"]


def test_empty_query_is_rejected(monkeypatch):
    import pytest

    embedder, _ = make_embedder(monkeypatch)

    with pytest.raises(ValueError):
        embedder.embed_query("   ")
