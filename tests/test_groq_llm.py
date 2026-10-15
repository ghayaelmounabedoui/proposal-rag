import pytest

from proposal_rag.generation import groq_llm
from proposal_rag.generation.groq_llm import EmptyLLMResponse, GroqLLM


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0
        self.kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        content = self.contents.pop(0)
        message = type("M", (), {"content": content})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice]})()


def make_llm(monkeypatch, contents):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    llm = GroqLLM()
    completions = FakeCompletions(contents)
    llm.client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    return llm, completions


def test_returns_stripped_content(monkeypatch):
    llm, completions = make_llm(monkeypatch, ["  bonjour  "])

    assert llm.generate("sys", "user") == "bonjour"
    assert completions.kwargs["max_completion_tokens"] == llm.max_tokens


def test_retries_after_empty_response(monkeypatch):
    llm, completions = make_llm(monkeypatch, [None, "  ", "enfin"])

    assert llm.generate("sys", "user") == "enfin"
    assert completions.calls == 3


def test_raises_after_too_many_empty_responses(monkeypatch):
    llm, completions = make_llm(monkeypatch, [None] * groq_llm.MAX_ATTEMPTS)

    with pytest.raises(EmptyLLMResponse):
        llm.generate("sys", "user")

    assert completions.calls == groq_llm.MAX_ATTEMPTS


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError):
        GroqLLM()
