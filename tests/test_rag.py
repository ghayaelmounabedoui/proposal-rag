import pytest

from proposal_rag.generation.context import ContextBuilder
from proposal_rag.rag import ProposalRAG, format_source

from .conftest import FakeLLM, FakeRetrieval, make_result


def build_rag(results, **kwargs):
    llm = FakeLLM()
    rag = ProposalRAG(
        llm=llm,
        retrieval=FakeRetrieval(results),
        **kwargs,
    )
    return rag, llm


def test_answer_includes_sources_in_prompt_order():
    rag, llm = build_rag(
        [
            make_result(source="a.md", chunk_id="1"),
            make_result(source="b.md", chunk_id="2"),
        ]
    )

    result = rag.ask_with_sources("Question ?", top_k=2)

    assert result["answer"] == "Réponse de test [SOURCE 1]"
    assert [s["id"] for s in result["sources"]] == [
        "[SOURCE 1]",
        "[SOURCE 2]",
    ]
    assert [s["document"] for s in result["sources"]] == ["a.md", "b.md"]
    assert "[SOURCE 2]" in llm.calls[0][1]


def test_no_results_does_not_call_llm():
    rag, llm = build_rag([])

    result = rag.ask_with_sources("Question ?")

    assert result["sources"] == []
    assert "do not provide enough information" in result["answer"]
    assert llm.calls == []


def test_sources_only_list_context_actually_sent():
    long_text = "mot " * 200
    rag, _ = build_rag(
        [
            make_result(text=long_text, chunk_id="1"),
            make_result(text=long_text, chunk_id="2"),
        ],
        context_builder=ContextBuilder(max_chars=900),
    )

    result = rag.ask_with_sources("Question ?", top_k=2)

    assert len(result["sources"]) == 1


def test_empty_question_rejected():
    rag, _ = build_rag([make_result()])

    with pytest.raises(ValueError):
        rag.ask("   ")


def test_format_source_handles_bad_heading_and_truncates():
    result = make_result(text="x" * 1000, heading="not json")
    source = format_source(1, result)

    assert source["section"] == "not json"
    assert len(source["excerpt"]) == 300
