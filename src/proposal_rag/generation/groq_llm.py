import logging
import os

from groq import Groq

from proposal_rag.generation.llm import LLM

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Le modèle raisonne avant de répondre : un budget trop petit peut être
# consommé entièrement par le raisonnement et donner une réponse vide.
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "8000"))
MAX_ATTEMPTS = 3


class EmptyLLMResponse(RuntimeError):
    """Le modèle n'a produit aucun texte après plusieurs tentatives."""


class GroqLLM(LLM):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY n'est pas définie."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_completion_tokens=self.max_tokens,
            )

            choice = response.choices[0]
            content = choice.message.content

            if content and content.strip():
                return content.strip()

            logger.warning(
                "Réponse vide de Groq (tentative %d/%d, finish_reason=%s)",
                attempt,
                MAX_ATTEMPTS,
                choice.finish_reason,
            )

        raise EmptyLLMResponse(
            "Groq a retourné une réponse vide après "
            f"{MAX_ATTEMPTS} tentatives."
        )
