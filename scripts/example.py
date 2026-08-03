# scripts/example.py
# -*- coding: utf-8 -*-
"""End-to-end example for Directed Evolution in Codon Space.

Demonstrates three modes on a single parental CDS:

  1. Evo scoring          - synonymous-constrained AND unconstrained.
  2. Single-point scan    - every preferred synonymous single-point
                            mutation on the parental background
                            (Hie-et-al.-style deep-mutational-scan
                            readout, in codon space).
  3. Directed evolution   - the greedy iterative walk from the paper,
                            under a 10% mutation cap.

Model weights, tokenizer, and config are pulled from Hugging Face
automatically the first time you run this.
"""

from .evolve import Evolver
from .codon_utils import print_optimization_report, print_mutation_scan


# -------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------

HOST_TOKEN_TYPE = 345   # C. griseus (CHO). See species_token_type.py for the full list.

# Parental CDS - replace with your own. This is just a short toy heavy
# chain signal peptide + a couple codons + stop so the example runs fast
# on CPU. Real usage: paste your full ~450-codon CDS here.
PARENTAL_CDS = (
    "ATGGACTGGACCTGGCGCATCCTGTTCCTGGTGGCCGCCGCCACCGGCGCCCACTCCTAA"
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
    # 1. Evo scoring (both flavors)
    # ---------------------------------------------------------------
    section("1. Evo scoring")
    score = evolver.evo_score(PARENTAL_CDS, token_type_id=HOST_TOKEN_TYPE)
    print(f"Length (codons)                               : "
          f"{len(PARENTAL_CDS) // 3}")
    print(f"Evo score - synonymous-constrained (nats)     : "
          f"{score.synonymous_constrained:+.6f}")
    print(f"Evo score - unconstrained          (nats)     : "
          f"{score.unconstrained:+.6f}")
    print()
    print("Notes:")
    print("  - synonymous_constrained is the objective the directed-")
    print("    evolution walk climbs. Higher = more host-like codon")
    print("    choices, holding the protein fixed.")
    print("  - unconstrained is normalized over the full codon vocab.")
    print("    It is sensitive to amino-acid identity, so use it as a")
    print("    global-plausibility sanity check and for cross-protein")
    print("    comparisons of similar length.")

    # ---------------------------------------------------------------
    # 2. Single-point synonymous mutation scan
    # ---------------------------------------------------------------
    section("2. Preferred single-point synonymous mutations (parental background)")
    scan = evolver.scan_mutations(
        PARENTAL_CDS,
        token_type_id=HOST_TOKEN_TYPE,
        min_delta_nats=0.0,     # report every strictly-preferred swap
        sort=True,              # sort by delta descending
    )

    if not scan:
        print("(No positions where the model prefers a different synonym.)")
    else:
        print(f"{len(scan)} preferred single-point synonymous mutations found.\n")
        # Full ranked table:
        print_mutation_scan(scan)

        # Practical downstream use: top-N recipe you'd hand to
        # a wet-lab collaborator for site-directed mutagenesis.
        top_n = min(5, len(scan))
        print(f"\nTop-{top_n} single-point recipe (highest log-odds Delta):")
        for rank, m in enumerate(scan[:top_n], start=1):
            print(f"  {rank}. pos {m.pos_1based:>4}  ({m.aa})  "
                  f"{m.from_codon} -> {m.to_codon}   "
                  f"Delta = {m.delta_nats:+.4f} nats  "
                  f"(odds x {m.odds_ratio:.2f})")

        print("\nNote: every Delta above is computed on the PARENTAL background")
        print("with no context updates between positions. For a trajectory that")
        print("accounts for how each edit reshapes the model's preferences at")
        print("neighboring positions, use directed evolution (section 3).")

    # ---------------------------------------------------------------
    # 3. Directed-evolution walk (the paper's main procedure)
    # ---------------------------------------------------------------
    section("3. Directed evolution in codon space (10% mutation cap)")
    results = evolver.optimize(
        [PARENTAL_CDS],
        token_type_id=HOST_TOKEN_TYPE,
        max_change_fraction=0.10,   # paper default
    )
    print_optimization_report(results)

    # ---------------------------------------------------------------
    # Cross-mode comparison
    # ---------------------------------------------------------------
    section("Cross-mode consistency check")
    if scan and results[0].trace:
        top_scan_hit = scan[0]
        first_walk_step = results[0].trace[0]
        print("Highest-Delta hit from single-point scan (parental background):")
        print(f"  pos {top_scan_hit.pos_1based}  "
              f"{top_scan_hit.from_codon} -> {top_scan_hit.to_codon}   "
              f"Delta = {top_scan_hit.delta_nats:+.4f} nats")
        print("First edit accepted by the greedy walk:")
        print(f"  pos {first_walk_step.pos_1based}  "
              f"{first_walk_step.from_codon} -> {first_walk_step.to_codon}   "
              f"Delta = {first_walk_step.delta_nats:+.4f} nats")
        agree = (
            top_scan_hit.pos_1based == first_walk_step.pos_1based
            and top_scan_hit.to_codon == first_walk_step.to_codon
        )
        print(f"Agree on iteration 0? {'YES' if agree else 'NO'}")
        print("(Should be YES: the greedy walk's first move is exactly the")
        print(" highest-Delta hit on the parental background. Subsequent")
        print(" walk moves may diverge from the scan because context updates.)")


if __name__ == "__main__":
    main()