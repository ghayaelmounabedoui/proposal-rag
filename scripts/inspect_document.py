import argparse
from pathlib import Path

from proposal_rag.ingestion.chunker import chunk_document
from proposal_rag.ingestion.loader import load_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Afficher les chunks d'un document."
    )
    parser.add_argument("document", type=Path)
    args = parser.parse_args()

    blocks = load_document(args.document)
    chunks = chunk_document(blocks, source=args.document.name)

    print(f"Document : {args.document.name}")
    print(f"Blocs : {len(blocks)}")
    print(f"Chunks : {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n{'=' * 60}")
        print(f"CHUNK {index}")
        print(chunk.text)


if __name__ == "__main__":
    main()
