# codon_utils.py
# -*- coding: utf-8 -*-
"""
Shared utilities for codon-level language-model optimization:
- Standard genetic code (AA -> synonymous codons)
- DNA cleaning / codon splitting
- Result dataclasses (optimization walk + single-point scan + dual evo score)
- Reporting helpers for pretty-printing results
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Sequence


# -------------------- Standard genetic code (DNA alphabet) --------------------

DEFAULT_SYNONYMOUS_CODONS = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAT", "AAC"],
    "D": ["GAT", "GAC"],
    "C": ["TGT", "TGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "K": ["AAA", "AAG"],
    "M": ["ATG"],
    "F": ["TTT", "TTC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "*": ["TAA", "TAG", "TGA"],
}


# -------------------- Sequence helpers --------------------

def clean_dna(seq):
    """Uppercase, strip whitespace, and convert RNA (U) to DNA (T)."""
    if not isinstance(seq, str):
        return ""
    return "".join(seq.split()).upper().replace("U", "T")


def dna_to_codons(seq):
    """Split a DNA string into a list of length-3 codons."""
    if len(seq) % 3 != 0:
        raise ValueError(f"Sequence length {len(seq)} is not a multiple of 3.")
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]


def build_codon_to_aa(synonymous_codons):
    """Reverse mapping: codon -> amino acid."""
    return {c: aa for aa, cs in synonymous_codons.items() for c in cs}


def hash_ids(ids):
    """Stable SHA1 hash of a token-id sequence (used for cycle detection)."""
    return hashlib.sha1(",".join(map(str, ids)).encode("utf-8")).hexdigest()


# -------------------- Result dataclasses --------------------

@dataclass
class ChangeStep:
    """One accepted synonymous swap during greedy optimization."""
    iter: int
    pos_1based: int
    from_codon: str
    to_codon: str
    runner_up_codon: str
    delta_nats: float
    odds_ratio: float
    margin_to_2nd_nats: float
    changed_positions_vs_orig: int
    change_cap_allowed: int


@dataclass
class SingleMutation:
    """One preferred synonymous single-point mutation (unapplied).

    Produced by Evolver.scan_mutations. This is the codon-space analogue
    of the per-residue deep-mutational-scan output in efficient evolution
    of antibodies (Hie et al., Nat Biotechnol 2024): mask a position, get
    the model's preferences, keep the best synonymous swap. All deltas
    are computed on the parental background - no context updates between
    positions and no edits applied.
    """
    pos_1based: int
    aa: str
    from_codon: str
    to_codon: str
    runner_up_codon: str
    delta_nats: float           # log P(to) - log P(from) under synonym-restricted softmax
    odds_ratio: float           # exp(delta_nats)
    margin_to_2nd_nats: float   # log P(to) - log P(runner_up)


@dataclass
class EvoScore:
    """Evo score reported both with and without the synonymous constraint.

    synonymous_constrained
        Mean log-softmax at each position, normalized over the synonyms
        of the encoded amino acid. This is the objective the directed-
        evolution walk climbs - a measure of how host-like the codon
        choice is, holding the protein fixed.

    unconstrained
        Mean log-softmax at each position, normalized over the full
        codon vocabulary. Sensitive to amino-acid identity, so it is
        comparable across proteins of similar length and useful for
        cross-checking that a synonymous rewrite has not drifted into a
        globally-unlikely regime.

    Both values are in nats. Higher = more plausible under the host LM.
    """
    synonymous_constrained: float
    unconstrained: float

    def __float__(self):
        # Back-compat: float(evolver.evo_score(...)) returns the
        # synonymous-constrained score, which is what the walk optimizes.
        return self.synonymous_constrained

    def __repr__(self):
        return (
            f"EvoScore(synonymous_constrained={self.synonymous_constrained:.6f}, "
            f"unconstrained={self.unconstrained:.6f})"
        )


@dataclass
class OptimizationResult:
    """Full result for a single optimized sequence."""
    input_index: int
    original_dna: str
    optimized_dna: str
    length_codons: int
    positions_changed: int
    original_evo_score: float                    # synonymous-constrained (back-compat)
    optimized_evo_score: float                   # synonymous-constrained (back-compat)
    delta_evo_score: float                       # synonymous-constrained delta
    stop_reason: str
    trace: list = field(default_factory=list)
    # New: keep both flavors of the evo score for downstream analysis / plotting.
    original_evo_score_unconstrained: float = 0.0
    optimized_evo_score_unconstrained: float = 0.0


# -------------------- Reporting --------------------

def print_optimization_report(results):
    """Pretty-print original vs new evo scores and every preferred change."""
    for r in results:
        header = f" Sequence #{r.input_index}  (L = {r.length_codons} codons) "
        bar = "=" * len(header)
        print(bar)
        print(header)
        print(bar)
        print(f"Original  evo score (syn)      : {r.original_evo_score:+.6f} nats")
        print(f"Optimized evo score (syn)      : {r.optimized_evo_score:+.6f} nats")
        print(
            f"Delta evo score (syn)          : {r.delta_evo_score:+.6f} nats "
            f"(odds x {math.exp(r.delta_evo_score):.3f})"
        )
        print(
            f"Original  evo score (unconstr.): "
            f"{r.original_evo_score_unconstrained:+.6f} nats"
        )
        print(
            f"Optimized evo score (unconstr.): "
            f"{r.optimized_evo_score_unconstrained:+.6f} nats"
        )
        pct = 100.0 * r.positions_changed / max(r.length_codons, 1)
        print(
            f"Positions changed              : "
            f"{r.positions_changed} / {r.length_codons} ({pct:.1f}%)"
        )
        print(f"Stop reason                    : {r.stop_reason}")

        if r.trace:
            print(f"\nPreferred changes ({len(r.trace)}):")
            print(
                f"  {'iter':>4}   {'pos':>4}   from -> to     "
                f"{'delta (nats)':>12}   {'odds':>6}   runner-up"
            )
            for s in r.trace:
                print(
                    f"  {s.iter:>4}   {s.pos_1based:>4}   "
                    f"{s.from_codon} -> {s.to_codon}     "
                    f"{s.delta_nats:>12.4f}   {s.odds_ratio:>6.3f}   "
                    f"{s.runner_up_codon}"
                )
        print()


def print_mutation_scan(scan, top=None):
    """Pretty-print a ranked list of preferred single-point synonymous mutations.

    Parameters
    ----------
    scan : sequence of SingleMutation
        Output of Evolver.scan_mutations.
    top : int, optional
        If given, only print the top-N by delta. Default: print all.
    """
    items = sorted(scan, key=lambda m: -m.delta_nats)
    if top is not None:
        items = items[:top]
    if not items:
        print("(no preferred synonymous mutations above threshold)")
        return
    print(
        f"{'pos':>6} {'aa':>3} {'from':>5}    {'to':<5} "
        f"{'delta (nats)':>12} {'odds':>7}   runner-up"
    )
    for m in items:
        print(
            f"{m.pos_1based:>6} {m.aa:>3} {m.from_codon:>5} -> {m.to_codon:<5} "
            f"{m.delta_nats:>12.4f} {m.odds_ratio:>7.3f}   {m.runner_up_codon}"
        )