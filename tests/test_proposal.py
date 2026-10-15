
import pytest

from proposal_rag.generation.llm import LLM
from proposal_rag.rfp.proposal_generator import ProposalGenerator


class FakeLLM(LLM):
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "# Technical Proposal\n\nTest proposal."


def test_generate_proposal():
    llm = FakeLLM()
    generator = ProposalGenerator(llm=llm)

    matrix = [
        {
            "id": "REQ-001",
            "requirement": "Data must use AES-256 encryption.",
            "status": "COMPLIANT",
            "answer": "AES-256 is documented.",
            "evidence": ["[SOURCE 1]"],
        }
    ]

    result = generator.generate(matrix)

    assert "# Technical Proposal" in result
    assert "AES-256" in llm.user_prompt
    assert "COMPLIANT" in llm.user_prompt
    assert "[SOURCE 1]" in llm.user_prompt


def test_generate_proposal_empty_matrix():
    generator = ProposalGenerator(llm=FakeLLM())

    with pytest.raises(ValueError, match="matrice est vide"):
        generator.generate([])
