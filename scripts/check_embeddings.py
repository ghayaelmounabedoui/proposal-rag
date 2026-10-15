from pathlib import Path

from proposal_rag.embeddings.local import LocalEmbedder
from proposal_rag.ingestion.chunker import chunk_document
from proposal_rag.ingestion.loader import load_document


def main() -> None:
    path = Path("data/raw/security.md")

    blocks = load_document(path)
    chunks = chunk_document(blocks, source=path.name)

    if not chunks:
        raise ValueError("Le document ne contient aucun paragraphe.")

    embedder = LocalEmbedder()
    # Un seul chunk pour vérifier l'intégration.
    document_vector = embedder.embed_documents(
        [chunks[0].text]
    )[0]

    query_vector = embedder.embed_query(
        "Comment protégez-vous les données ?"
    )

    print("Chunk envoyé :")
    print(chunks[0].text)
    print()
    print(f"Dimensions document : {len(document_vector)}")
    print(f"Dimensions question : {len(query_vector)}")
    print(f"Premières valeurs : {document_vector[:5]}")


if __name__ == "__main__":
    main()
