"""Fixtures partagées : aucun modèle ni appel réseau n'est utilisé."""

import os

import pytest
from fastapi.testclient import TestClient

# Doit être défini avant l'import de l'API.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.pop("API_KEY", None)

from proposal_rag.generation.llm import LLM  # noqa: E402
from proposal_rag.retrieval.models import RetrievalResult  # noqa: E402


class FakeLLM(LLM):
    def __init__(self, reply: str = "Réponse de test [SOURCE 1]"):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.reply


class FakeRetrieval:
    """Remplace RetrievalPipeline sans charger de modèle."""

    def __init__(self, results: list[RetrievalResult] | None = None):
        self.results = results or []
        self.refresh_count = 0
        self.hybrid_retriever = type(
            "H",
            (),
            {
                "dense_retriever": type(
                    "D", (), {"embedder": object(), "store": object()}
                )()
            },
        )()

    def search(self, query, top_k=5, candidate_k=10):
        return self.results[:top_k]

    def refresh(self):
        self.refresh_count += 1


def make_result(
    text="Les données sont chiffrées avec AES-256.",
    source="security.md",
    heading='["Sécurité", "Chiffrement"]',
    score=0.9,
    chunk_id="c1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={
            "source": source,
            "heading_path": heading,
            "content": text,
        },
    )


class FakeRAG:
    def __init__(self):
        self.retrieval = FakeRetrieval([make_result()])
        self.generator = type("G", (), {"llm": FakeLLM()})()

    def ask_with_sources(self, question, top_k, candidate_k):
        from proposal_rag.rag import format_source

        return {
            "answer": f"Réponse à : {question}",
            "sources": [format_source(1, self.retrieval.results[0])],
        }


@pytest.fixture
def fake_rag():
    return FakeRAG()


@pytest.fixture
def client(monkeypatch, tmp_path, fake_rag):
    import api.main as main

    monkeypatch.setattr(main, "ProposalRAG", lambda llm: fake_rag)
    monkeypatch.setattr(main, "GroqLLM", lambda: FakeLLM())
    monkeypatch.setattr(main, "RAW_DIR", tmp_path / "raw")

    with TestClient(main.app) as test_client:
        yield test_client
