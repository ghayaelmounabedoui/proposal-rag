from proposal_rag.retrieval.models import RetrievalResult


class ContextBuilder:
    def __init__(
        self,
        max_chars: int = 12_000,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars doit être supérieur à 0.")

        self.max_chars = max_chars

    def build(
        self,
        results: list[RetrievalResult],
    ) -> str:
        return self.build_with_sources(results)[0]

    def build_with_sources(
        self,
        results: list[RetrievalResult],
    ) -> tuple[str, list[RetrievalResult]]:
        """Construire le contexte et renvoyer les sources réellement
        incluses, dans l'ordre des identifiants [SOURCE X]."""
        if not results:
            return "", []

        parts: list[str] = []
        used: list[RetrievalResult] = []
        current_length = 0

        for index, result in enumerate(results, start=1):
            metadata = result.metadata or {}

            source = metadata.get("source", "unknown")
            heading = metadata.get("heading_path")

            header = f"[SOURCE {index}]\nDocument: {source}"

            if heading:
                header += f"\nSection: {heading}"

            block = (
                f"{header}\n\n"
                f"{result.text.strip()}"
            )

            # +2 correspond aux deux retours à la ligne
            # ajoutés entre les sources.
            additional_length = len(block)

            if parts:
                additional_length += 2

            if current_length + additional_length > self.max_chars:
                break

            parts.append(block)
            used.append(result)
            current_length += additional_length

        return "\n\n".join(parts), used