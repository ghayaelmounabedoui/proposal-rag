import argparse
from pathlib import Path

from proposal_rag.embeddings.local import LocalEmbedder
from proposal_rag.indexes.chroma_store import ChromaStore
from proposal_rag.ingestion.chunker import chunk_document
from proposal_rag.ingestion.loader import load_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexer un document dans ChromaDB."
    )
    parser.add_argument("document", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    corpus_root = project_root / "data" / "raw"
    path = args.document.resolve()

    # Utiliser un chemin relatif stable pour identifier le document.
    try:
        source = path.relative_to(corpus_root).as_posix()
    except ValueError:
        raise ValueError(
            "Place le document dans data/raw avant de l'indexer."
        ) from None

    blocks = load_document(path)
    chunks = chunk_document(blocks, source=source)

    if not chunks:
        raise ValueError("Aucun paragraphe à indexer.")

    print(f"Document : {source}")
    print(f"Chunks : {len(chunks)}")
    print("Calcul des embeddings…")

    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents(
        [chunk.text for chunk in chunks]
    )

    store = ChromaStore()
    store.upsert_chunks(chunks, embeddings)

    print("Indexation terminée.")
    print(f"Total dans la collection : {store.count()}")


if __name__ == "__main__":
    main()
