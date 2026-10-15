import argparse

from proposal_rag.retrieval.hybrid import build_hybrid_retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recherche hybride Dense + BM25."
    )

    parser.add_argument(
        "question",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    retriever = build_hybrid_retriever()

    results = retriever.search(
        query=args.question,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    print(f"\nQuestion : {args.question}")

    if not results:
        print("\nAucun résultat trouvé.")
        return

    for rank, result in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"Résultat {rank}")
        print("=" * 70)

        print(f"Chunk ID : {result.chunk_id}")

        source = result.metadata.get("source")
        if source:
            print(f"Document : {source}")

        heading = result.metadata.get("heading_path")
        if heading:
            print(f"Section : {heading}")

        print(f"Score RRF : {result.score:.6f}")

        print("\nPassage :")
        print(result.text)


if __name__ == "__main__":
    main()
