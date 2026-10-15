
from proposal_rag.generation.context import ContextBuilder
from proposal_rag.retrieval.models import RetrievalResult


def test_context_contains_sources():
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            text="Data is encrypted using AES-256.",
            score=0.95,
            metadata={
                "source": "security.md",
                "heading_path": "Encryption",
            },
        )
    ]

    builder = ContextBuilder(max_chars=2000)
    context = builder.build(results)

    assert "[SOURCE 1]" in context
    assert "security.md" in context
    assert "AES-256" in context


def test_empty_context():
    builder = ContextBuilder()

    assert builder.build([]) == ""
