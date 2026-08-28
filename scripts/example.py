# scripts/example.py
# -*- coding: utf-8 -*-
"""End-to-end example for Directed Evolution in Codon Space.

Runs three steps on a single parental CDS:
  1. Evo score
  2. Single-point synonymous mutation scan
  3. Directed evolution (greedy walk, 10% mutation cap)

Model weights, tokenizer, and config are pulled from Hugging Face
automatically the first time you run this.
"""

from evolve import Evolver
from codon_utils import print_optimization_report, print_mutation_scan


# -------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------

HOST_TOKEN_TYPE = 345   # C. griseus (CHO). See species_token_type.py for the full list.

# Parental CDS - replace with your own full-length coding sequence.
PARENTAL_CDS = (
    "ATCCAGCTGACCCAGAGCCCCAGCAGCCTGAGCGCCAGCGTGGGCGACCGGGTG"
)


def section(title):
    """Print a visual section divider."""
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def main():
    # ---------------------------------------------------------------
    # Load SynCodonLM v2 from Hugging Face
    # ---------------------------------------------------------------
    section("Loading SynCodonLM v2 (Hugging Face)")
    evolver = Evolver()   # downloads + caches on first use

    # ---------------------------------------------------------------
    # 1. Evo scoring
    # ---------------------------------------------------------------
    section("1. Evo scoring")
    score = evolver.evo_score(PARENTAL_CDS, token_type_id=HOST_TOKEN_TYPE)
    print(f"Length (codons)                : {len(PARENTAL_CDS) // 3}")
    print(f"Evo score (synonymous-constrained) : {score.synonymous_constrained:+.4f} nats")
    print(f"Evo score (unconstrained)          : {score.unconstrained:+.4f} nats")

    # ---------------------------------------------------------------
    # 2. Single-point synonymous mutation scan
    # ---------------------------------------------------------------
    section("2. Preferred single-point synonymous mutations")
    scan = evolver.scan_mutations(
        PARENTAL_CDS,
        token_type_id=HOST_TOKEN_TYPE,
        min_delta_nats=0.0,
        sort=True,
    )

    if not scan:
        print("No positions where the model prefers a different synonym.")
    else:
        print(f"{len(scan)} preferred single-point mutations found.\n")
        print_mutation_scan(scan)

        top_n = min(5, len(scan))
        print(f"\nTop-{top_n} single-point recipe:")
        for rank, m in enumerate(scan[:top_n], start=1):
            print(f"  {rank}. pos {m.pos_1based:>4}  ({m.aa})  "
                  f"{m.from_codon} -> {m.to_codon}   "
                  f"delta = {m.delta_nats:+.4f} nats")

    # ---------------------------------------------------------------
    # 3. Directed-evolution walk
    # ---------------------------------------------------------------
    section("3. Directed evolution (10% mutation cap)")
    results = evolver.optimize(
        [PARENTAL_CDS],
        token_type_id=HOST_TOKEN_TYPE,
        max_change_fraction=0.10,
    )
    result = results[0] #results is a list, as it can take numerous sequences in one pass
    print_optimization_report(results)
    print(f"Here is your optimized DNA: {result.optimized_dna}")


if __name__ == "__main__":
    main()
