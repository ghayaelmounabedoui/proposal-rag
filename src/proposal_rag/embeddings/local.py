from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self) -> None:
        self.model_name = "intfloat/multilingual-e5-small"

        self.model = SentenceTransformer(
            self.model_name,
            device="cpu",
        )

    def _encode(
        self,
        texts: list[str],
        prefix: str,
    ) -> list[list[float]]:
        if not texts:
            return []

        prepared = []

        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"Texte vide à la position {index}.")

            value = f"{prefix}: {text}"

            # Refuser les passages trop longs plutôt que les tronquer.
            tokens = self.model.tokenizer(
                value,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            if len(tokens) > self.model.max_seq_length:
                raise ValueError(
                    f"Texte {index} trop long : {len(tokens)} tokens. "
                    f"Limite : {self.model.max_seq_length}. "
                    "Le paragraphe doit être traité avant indexation."
                )

            prepared.append(value)

        vectors = self.model.encode(
            prepared,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vectors.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._encode(texts, prefix="passage")

    def embed_query(
    self,
    query: str,
) -> list[float]:
        query = query.strip()

        if not query:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        # E5 est entraîné avec deux préfixes distincts : "query: " pour les
        # questions et "passage: " pour les documents indexés.
        return self._encode([query], prefix="query")[0]