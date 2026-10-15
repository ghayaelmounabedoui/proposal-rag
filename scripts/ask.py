
import argparse

from proposal_rag.generation.groq_llm import GroqLLM
from proposal_rag.rag import ProposalRAG


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG pour répondre aux appels d'offres."
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question à poser au RAG.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    print(f"\nQuestion : {args.question}")
    print("Recherche des passages pertinents...")

    llm = GroqLLM()
    rag = ProposalRAG(llm=llm)

    answer = rag.ask(
        question=args.question,
        top_k=args.top_k,
        candidate_k=10,
    )

    print("\n" + "=" * 70)
    print("RÉPONSE RAG")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()
