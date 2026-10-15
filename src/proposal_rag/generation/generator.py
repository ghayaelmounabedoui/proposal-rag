from proposal_rag.generation.llm import LLM
from proposal_rag.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)


class AnswerGenerator:
    def __init__(
        self,
        llm: LLM,
    ) -> None:
        self.llm = llm

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        question = question.strip()
        context = context.strip()

        if not question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        if not context:
            return (
                "The available documents do not provide "
                "enough information to answer this question."
            )

        user_prompt = build_user_prompt(
            question=question,
            context=context,
        )

        return self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
