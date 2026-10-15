import json

from proposal_rag.generation.context import ContextBuilder
from proposal_rag.generation.generator import AnswerGenerator
from proposal_rag.generation.llm import LLM
from proposal_rag.retrieval.models import RetrievalResult
from proposal_rag.retrieval.pipeline import RetrievalPipeline

EXCERPT_CHARS = 300


def format_source(index: int, result: RetrievalResult) -> dict:
    """Décrire une source citée ([SOURCE index]) pour l'API."""

    metadata = result.metadata or {}

    # Le chemin de titres est stocké en JSON dans ChromaDB.
    raw_heading = metadata.get("heading_path")
    try:
        heading = json.loads(raw_heading) if raw_heading else []
    except (TypeError, ValueError):
        heading = [str(raw_heading)]

    content = metadata.get("content") or result.text

    return {
        "id": f"[SOURCE {index}]",
        "document": metadata.get("source", "unknown"),
        "section": " > ".join(heading),
        "page": metadata.get("page_start"),
        "score": round(result.score, 4),
        "excerpt": content.strip()[:EXCERPT_CHARS],
    }


class ProposalRAG:
    def __init__(
        self,
        llm: LLM,
        retrieval: RetrievalPipeline | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.retrieval = (
            retrieval
            or RetrievalPipeline()
        )

        self.context_builder = (
            context_builder
            or ContextBuilder()
        )

        self.generator = AnswerGenerator(llm)

    def ask_with_sources(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> dict:
        """Répondre à la question et renvoyer les sources fournies au LLM."""

        question = question.strip()

        if not question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        results = self.retrieval.search(
            query=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        context, used = self.context_builder.build_with_sources(
            results
        )

        answer = self.generator.generate(
            question=question,
            context=context,
        )

        return {
            "answer": answer,
            "sources": [
                format_source(index, result)
                for index, result in enumerate(used, start=1)
            ],
        }

    def ask(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> str:
        return self.ask_with_sources(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )["answer"]
