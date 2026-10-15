"""Mesurer la qualité de la recherche sur un jeu de questions de référence.

Chaque question indique le document qui doit apparaître dans les résultats.
Métriques : hit@1, hit@k (le bon document est dans les k premiers) et MRR.

Exemple :
    python scripts/evaluate.py --k 3 --min-hit-rate 0.8
"""

import argparse
import json
import sys
from pathlib import Path

from proposal_rag.retrieval.pipeline import RetrievalPipeline

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.0,
        help="Code de sortie 1 si hit@k est inférieur à ce seuil.",
    )
    args = parser.parse_args()

    questions = json.loads(args.dataset.read_text(encoding="utf-8"))
    pipeline = RetrievalPipeline()

    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_ranks = 0.0

    for item in questions:
        results = pipeline.search(
            query=item["question"],
            top_k=args.k,
            candidate_k=10,
        )
        sources = [r.metadata.get("source") for r in results]

        rank = next(
            (
                position
                for position, source in enumerate(sources, start=1)
                if source == item["expected_source"]
            ),
            None,
        )

        hit_at_1 += rank == 1
        hit_at_k += rank is not None
        reciprocal_ranks += 1 / rank if rank else 0.0

        status = f"rang {rank}" if rank else "ABSENT"
        print(f"[{status:>8}] {item['question']}")
        if rank != 1:
            print(f"           attendu : {item['expected_source']} | obtenu : {sources}")

    total = len(questions)
    hit_rate = hit_at_k / total

    print()
    print(f"Questions : {total}")
    print(f"hit@1     : {hit_at_1 / total:.0%}")
    print(f"hit@{args.k}     : {hit_rate:.0%}")
    print(f"MRR       : {reciprocal_ranks / total:.3f}")

    return 0 if hit_rate >= args.min_hit_rate else 1


if __name__ == "__main__":
    sys.exit(main())
