from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise NotImplementedError
