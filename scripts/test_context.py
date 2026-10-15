import argparse

from proposal_rag.generation.context import ContextBuilder
from proposal_rag.retrieval.pipeline import RetrievalPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tester le retrieval et la construction du contexte."
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

    args = parser.parse_args()

    print(f"\nQuestion : {args.question}")
    print("\nRecherche des passages pertinents...")

    retrieval = RetrievalPipeline()

    results = retrieval.search(
        query=args.question,
        top_k=args.top_k,
        candidate_k=10,
    )

    print(f"Passages retenus : {len(results)}")

    builder = ContextBuilder()

    context = builder.build(results)

    print("\n" + "=" * 70)
    print("CONTEXTE ENVOYÉ AU LLM")
    print("=" * 70)

    if not context:
        print("Aucun contexte trouvé.")
        return

    print(context)


if __name__ == "__main__":
    main()
